- **Bedrock errors name the AWS exception** (`adk-model`): a `BedrockClient` request that AWS
  rejected, and an exception event inside a `ConverseStream` response, failed with
  `model.legacy` (`Internal`) and a message ending in `service error`. They now fail with
  `model.bedrock.api_error`, a message that quotes the exception name and up to 512 bytes of
  the AWS message, the name in `details.metadata["provider_error_code"]` (for example
  `ValidationException`), and the request id. A rejected request takes the category of its
  HTTP status, which `details.upstream_status_code` records, so model timeouts (408),
  throttling (429) and service errors (500, 503) are retryable and other 4xx rejections are
  not; `ThrottlingException` and `ModelNotReadyException` are always `RateLimited`. An
  exception event takes the category of the status AWS documents for it and records no
  upstream status; `modelStreamErrorException`, which AWS documents as one to retry, is
  `Unavailable`, and an undocumented one is `Internal`.
