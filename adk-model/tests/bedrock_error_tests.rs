//! Bedrock service errors name the AWS exception, quote a bounded excerpt of its message,
//! and take their category from the HTTP status.
#![cfg(feature = "bedrock")]

use adk_core::{AdkError, Content, ErrorCategory, Llm, LlmRequest, UPSTREAM_EXCERPT_BYTES};
use adk_model::bedrock::{BedrockClient, BedrockConfig};
use aws_sdk_bedrockruntime::config::retry::RetryConfig;
use aws_sdk_bedrockruntime::config::{Region, Token};
use futures::StreamExt;
use serde_json::{Value, json};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A client for `uri` with SDK retries off, so a 429 or 5xx stub is hit once without backoff.
fn client(uri: &str) -> BedrockClient {
    let sdk = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version_latest()
        .region(Region::new("us-east-1"))
        .bearer_token(Token::new("key", None))
        .endpoint_url(uri)
        .retry_config(RetryConfig::disabled())
        .build();
    BedrockClient::from_client(
        BedrockConfig::new("us-east-1", "m"),
        aws_sdk_bedrockruntime::Client::from_conf(sdk),
    )
}

/// A rejection carrying the AWS error-type header when `error_type` is given.
fn rejection(status: u16, error_type: Option<&str>, body: &str) -> ResponseTemplate {
    let response = ResponseTemplate::new(status)
        .insert_header("x-amzn-requestid", "req-1")
        .insert_header("content-type", "application/json")
        .set_body_string(body);
    match error_type {
        Some(error_type) => response.insert_header("x-amzn-errortype", error_type),
        None => response,
    }
}

/// A stub that answers one Converse or ConverseStream call with `response`.
async fn rejecting(response: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/m/converse(-stream)?$"))
        .respond_with(response)
        .expect(1)
        .mount(&server)
        .await;
    server
}

/// Runs one turn, streamed or not, and returns the error that ended it.
async fn failure(model: &BedrockClient, stream: bool) -> AdkError {
    let request = LlmRequest::new("m", vec![Content::new("user").with_text("Hello")]);
    match model.generate_content(request, stream).await {
        Err(error) => error,
        Ok(mut responses) => {
            responses.next().await.expect("stream yields an item").expect_err("request fails")
        }
    }
}

/// The parts of an error a caller acts on, with its details as JSON for whole-object equality.
fn summary(error: &AdkError) -> (ErrorCategory, &str, &str, Value) {
    let details = serde_json::to_value(&*error.details).expect("details serialize");
    (error.category, error.code, error.message.as_str(), details)
}

#[tokio::test]
async fn send_errors_take_the_category_of_the_http_status() {
    let cases = [
        (400, "ValidationException", ErrorCategory::InvalidInput),
        (403, "AccessDeniedException", ErrorCategory::Forbidden),
        (404, "ResourceNotFoundException", ErrorCategory::NotFound),
        (408, "ModelTimeoutException", ErrorCategory::Timeout),
        (424, "ModelErrorException", ErrorCategory::InvalidInput),
        (429, "ThrottlingException", ErrorCategory::RateLimited),
        (429, "ModelNotReadyException", ErrorCategory::RateLimited),
        (500, "InternalServerException", ErrorCategory::Unavailable),
        (503, "ServiceUnavailableException", ErrorCategory::Unavailable),
    ];
    for (status, name, category) in cases {
        let body = format!(r#"{{"message":"{name} from the stub"}}"#);
        for stream in [false, true] {
            let server = rejecting(rejection(status, Some(name), &body)).await;
            let error = failure(&client(&server.uri()), stream).await;
            let message = format!(
                "Bedrock API error (HTTP {status}) for region=us-east-1, model=m: {name}: {name} from the stub"
            );
            assert_eq!(
                summary(&error),
                (
                    category,
                    "model.bedrock.api_error",
                    message.as_str(),
                    json!({
                        "upstream_status_code": status,
                        "request_id": "req-1",
                        "provider": "bedrock",
                        "metadata": {"provider_error_code": name},
                    }),
                ),
                "{name} stream={stream}"
            );
            assert_eq!(
                error.is_retryable(),
                matches!(
                    category,
                    ErrorCategory::Timeout
                        | ErrorCategory::RateLimited
                        | ErrorCategory::Unavailable
                ),
                "{name} stream={stream}"
            );
        }
    }
}

#[tokio::test]
async fn throttling_is_rate_limited_whatever_the_status() {
    let body = r#"{"message":"Too many tokens"}"#;
    for stream in [false, true] {
        let server = rejecting(rejection(400, Some("ThrottlingException"), body)).await;
        let error = failure(&client(&server.uri()), stream).await;
        assert_eq!(
            summary(&error),
            (
                ErrorCategory::RateLimited,
                "model.bedrock.api_error",
                "Bedrock API error (HTTP 400) for region=us-east-1, model=m: ThrottlingException: Too many tokens",
                json!({
                    "upstream_status_code": 400,
                    "request_id": "req-1",
                    "provider": "bedrock",
                    "metadata": {"provider_error_code": "ThrottlingException"},
                }),
            ),
            "stream={stream}"
        );
        assert!(error.is_retryable(), "stream={stream}");
    }
}

#[tokio::test]
async fn the_error_type_header_is_reported_without_its_suffix_or_namespace() {
    let cases = [
        (
            400,
            "ValidationException:http://internal.amazon.com/coral/com.amazon.coral.validate/",
            "ValidationException",
            ErrorCategory::InvalidInput,
        ),
        (
            403,
            "aws.protocoltests#AccessDeniedException",
            "AccessDeniedException",
            ErrorCategory::Forbidden,
        ),
    ];
    for (status, header, name, category) in cases {
        let server = rejecting(rejection(status, Some(header), r#"{"message":"denied"}"#)).await;
        let error = failure(&client(&server.uri()), false).await;
        let message = format!(
            "Bedrock API error (HTTP {status}) for region=us-east-1, model=m: {name}: denied"
        );
        assert_eq!(
            summary(&error),
            (
                category,
                "model.bedrock.api_error",
                message.as_str(),
                json!({
                    "upstream_status_code": status,
                    "request_id": "req-1",
                    "provider": "bedrock",
                    "metadata": {"provider_error_code": name},
                }),
            ),
            "{header}"
        );
    }
}

#[tokio::test]
async fn unmodelled_and_unnamed_exceptions_keep_the_status_category() {
    let named = r#"{"message":"something new"}"#;
    let cases = [
        (400, Some("SomeNewException"), named, ErrorCategory::InvalidInput),
        (503, Some("SomeNewException"), named, ErrorCategory::Unavailable),
    ];
    for (status, header, body, category) in cases {
        for stream in [false, true] {
            let server = rejecting(rejection(status, header, body)).await;
            let error = failure(&client(&server.uri()), stream).await;
            let message = format!(
                "Bedrock API error (HTTP {status}) for region=us-east-1, model=m: SomeNewException: something new"
            );
            assert_eq!(
                summary(&error),
                (
                    category,
                    "model.bedrock.api_error",
                    message.as_str(),
                    json!({
                        "upstream_status_code": status,
                        "request_id": "req-1",
                        "provider": "bedrock",
                        "metadata": {"provider_error_code": "SomeNewException"},
                    }),
                ),
                "{status} stream={stream}"
            );
        }
    }

    // Without an exception name the SDK keeps no metadata, so a JSON message is dropped too.
    let unnamed = [
        (502, "<html>Bad Gateway</html>", ErrorCategory::Unavailable),
        (400, r#"{"message":"no name given"}"#, ErrorCategory::InvalidInput),
    ];
    for (status, body, category) in unnamed {
        for stream in [false, true] {
            let server = rejecting(rejection(status, None, body)).await;
            let error = failure(&client(&server.uri()), stream).await;
            let message = format!(
                "Bedrock API error (HTTP {status}) for region=us-east-1, model=m: unnamed exception"
            );
            assert_eq!(
                summary(&error),
                (
                    category,
                    "model.bedrock.api_error",
                    message.as_str(),
                    json!({
                        "upstream_status_code": status,
                        "request_id": "req-1",
                        "provider": "bedrock",
                    }),
                ),
                "{body} stream={stream}"
            );
        }
    }
}

#[tokio::test]
async fn the_service_message_is_quoted_up_to_the_excerpt_bound() {
    let long = "x".repeat(4 * UPSTREAM_EXCERPT_BYTES);
    let body = json!({"message": long}).to_string();
    let server = rejecting(rejection(400, Some("ValidationException"), &body)).await;
    let error = failure(&client(&server.uri()), false).await;
    let message = format!(
        "Bedrock API error (HTTP 400) for region=us-east-1, model=m: ValidationException: {}",
        "x".repeat(UPSTREAM_EXCERPT_BYTES)
    );
    assert_eq!(error.message, message);
}

#[tokio::test]
async fn a_validation_exception_is_sent_once_by_the_default_sdk_client() {
    let body = r#"{"message":"The model does not support cache points"}"#;
    let server = rejecting(rejection(400, Some("ValidationException"), body)).await;
    let model = BedrockClient::new_with_api_key(
        BedrockConfig::new("us-east-1", "m").with_endpoint_url(server.uri()),
        "key",
    )
    .unwrap();
    let error = failure(&model, false).await;
    assert_eq!(error.category, ErrorCategory::InvalidInput, "{error}");
    // Dropping the server verifies the stub saw exactly one request.
}
