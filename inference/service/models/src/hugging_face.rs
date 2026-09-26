pub(crate) fn revision_metadata_url(
    endpoint: &str,
    repository: &str,
    revision: &str,
) -> Result<reqwest::Url, String> {
    let mut url = reqwest::Url::parse(endpoint).map_err(|error| error.to_string())?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|()| "invalid Hugging Face endpoint".to_owned())?;
        segments.pop_if_empty().push("api").push("models");
        for segment in repository.split('/') {
            segments.push(segment);
        }
        segments.push("revision").push(revision);
    }
    Ok(url)
}

pub(crate) fn require_requested_revision(
    requested: &str,
    resolved: Option<&str>,
) -> Result<(), String> {
    match resolved {
        Some(resolved) if resolved == requested => Ok(()),
        Some(resolved) => Err(format!(
            "Hugging Face resolved requested revision {requested} as {resolved}",
        )),
        None => Err("Hugging Face metadata did not include a resolved revision".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_url_addresses_the_immutable_revision() {
        let revision = "a".repeat(40);
        let url = revision_metadata_url("https://huggingface.co/", "owner/repository", &revision)
            .unwrap();
        assert_eq!(
            url.as_str(),
            format!("https://huggingface.co/api/models/owner/repository/revision/{revision}"),
        );
    }

    #[test]
    fn resolved_revision_must_equal_the_requested_revision() {
        let requested = "a".repeat(40);
        assert!(require_requested_revision(&requested, Some(&requested)).is_ok());
        assert!(require_requested_revision(&requested, Some(&"b".repeat(40))).is_err());
        assert!(require_requested_revision(&requested, None).is_err());
    }
}
