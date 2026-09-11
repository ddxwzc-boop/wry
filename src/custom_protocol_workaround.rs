//! - WebView2 supports non-standard protocols only on Windows 10+, so we have to use a workaround.
//!   See <https://github.com/MicrosoftEdge/WebView2Feedback/issues/73>
//! - On Android, there's no API for registering custom protocols, so this workaround is also used.
//!
//! The process looks like this:
//!
//! 1. Use [`apply_uri_work_around`] to convert the URI we want to navigate to
//! 2. Intercept http(s) requests, test the request URI against [`is_work_around_uri`],
//!    if it matches, we apply [`revert_uri_work_around`] to the URI and feed it to the custom protocol handler

/// If the URI is a work around URI for this protocol which starts with `{http_or_https}://{protocol}.`
pub fn is_work_around_uri(uri: &str, http_or_https: &str, protocol: &str) -> bool {
  uri
    .strip_prefix(http_or_https)
    .and_then(|rest| rest.strip_prefix("://"))
    .and_then(|rest| rest.strip_prefix(protocol))
    .and_then(|rest| rest.strip_prefix("."))
    .is_some()
}

/// Conveting `{protocol}://localhost/abc` to `{http_or_https}://{protocol}.localhost/abc`
pub fn apply_uri_work_around(uri: &str, http_or_https: &str, protocol: &str) -> String {
  uri.replace(
    &original_uri_prefix(protocol),
    &work_around_uri_prefix(http_or_https, protocol),
  )
}

/// Conveting `{http_or_https}://{protocol}.localhost/abc` back to `{protocol}://localhost/abc`
pub fn revert_uri_work_around(uri: &str, http_or_https: &str, protocol: &str) -> String {
  uri.replace(
    &work_around_uri_prefix(http_or_https, protocol),
    &original_uri_prefix(protocol),
  )
}

pub fn original_uri_prefix(protocol: &str) -> String {
  format!("{protocol}://")
}

pub fn work_around_uri_prefix(http_or_https: &str, protocol: &str) -> String {
  format!("{http_or_https}://{protocol}.")
}

#[cfg(test)]
mod tests {
  use super::{
    apply_uri_work_around, is_work_around_uri, original_uri_prefix, revert_uri_work_around,
    work_around_uri_prefix,
  };

  #[test]
  fn checks_if_custom_protocol_uri() {
    let scheme = "http";
    let uri = "http://wry.localhost/path/to/page";
    assert!(is_work_around_uri(uri, scheme, "wry"));
    assert!(!is_work_around_uri(uri, scheme, "asset"));
  }

  #[test]
  fn https_work_around_roundtrip() {
    // OHOS uses https scheme for secure-context support
    let original = "tauri://localhost/index.html";
    let worked = apply_uri_work_around(original, "https", "tauri");
    assert_eq!(worked, "https://tauri.localhost/index.html");
    assert!(is_work_around_uri(&worked, "https", "tauri"));
    let reverted = revert_uri_work_around(&worked, "https", "tauri");
    assert_eq!(reverted, original);
  }

  #[test]
  fn https_work_around_does_not_match_external() {
    // External https URLs with an unrelated host must NOT be matched.
    // (Note: the upstream matcher is prefix-based, so a URL like
    // `https://tauri.com/` does match protocol `tauri` — that is upstream
    // behavior on Windows/Android and is kept as-is. OHOS uses this same
    // matcher in `dispatch_https_intercept_sync`; its ArkTS side applies the
    // equivalent gate by checking the first-dot host segment against the
    // seeded protocol set.)
    assert!(!is_work_around_uri("https://example.com/page", "https", "tauri"));
    // Only https://tauri.localhost/... matches
    assert!(is_work_around_uri("https://tauri.localhost/page", "https", "tauri"));
  }

  #[test]
  fn is_work_around_uri_rejects_non_http_scheme() {
    // A URL with a non-http(s) scheme should not match even if the host looks right
    assert!(!is_work_around_uri("ftp://wry.localhost/path", "http", "wry"));
    assert!(!is_work_around_uri("wry://localhost/path", "http", "wry"));
  }

  #[test]
  fn is_work_around_uri_rejects_wrong_protocol() {
    // URL registered for "wry" but checked against "asset"
    assert!(!is_work_around_uri("http://wry.localhost/path", "http", "asset"));
    // URL registered for "tauri" but checked against "wry"
    assert!(!is_work_around_uri("https://tauri.localhost/path", "https", "wry"));
  }

  #[test]
  fn is_work_around_uri_root_path() {
    // Root path (no trailing path) should still match
    assert!(is_work_around_uri("http://wry.localhost", "http", "wry"));
    assert!(is_work_around_uri("http://wry.localhost/", "http", "wry"));
  }

  #[test]
  fn apply_uri_work_around_noop_when_prefix_absent() {
    // If the URL doesn't start with `{protocol}://`, it's returned unchanged
    let url = "https://example.com/page";
    assert_eq!(apply_uri_work_around(url, "https", "tauri"), url);
  }

  #[test]
  fn revert_uri_work_around_noop_when_prefix_absent() {
    // If the URL doesn't contain the workaround prefix, it's returned unchanged
    let url = "tauri://localhost/index.html";
    assert_eq!(revert_uri_work_around(url, "https", "tauri"), url);
  }

  #[test]
  fn uri_work_around_preserves_query_and_fragment() {
    let original = "tauri://localhost/page?foo=bar#section";
    let worked = apply_uri_work_around(original, "https", "tauri");
    assert_eq!(
      worked,
      "https://tauri.localhost/page?foo=bar#section"
    );
    assert!(is_work_around_uri(&worked, "https", "tauri"));
    let reverted = revert_uri_work_around(&worked, "https", "tauri");
    assert_eq!(reverted, original);
  }

  #[test]
  fn original_uri_prefix_format() {
    assert_eq!(original_uri_prefix("tauri"), "tauri://");
    assert_eq!(original_uri_prefix("wry"), "wry://");
  }

  #[test]
  fn work_around_uri_prefix_format() {
    assert_eq!(
      work_around_uri_prefix("https", "tauri"),
      "https://tauri."
    );
    assert_eq!(work_around_uri_prefix("http", "wry"), "http://wry.");
  }

  #[test]
  fn apply_and_revert_are_inverses() {
    // Round-trip across multiple protocols
    for protocol in &["tauri", "wry", "asset", "custom"] {
      for scheme in &["http", "https"] {
        let original = format!("{}://localhost/path/to/resource", protocol);
        let worked = apply_uri_work_around(&original, scheme, protocol);
        assert!(is_work_around_uri(&worked, scheme, protocol));
        let reverted = revert_uri_work_around(&worked, scheme, protocol);
        assert_eq!(reverted, original);
      }
    }
  }
}
