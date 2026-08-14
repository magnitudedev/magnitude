use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use icn_contracts::models::{
    CatalogModelId, CatalogPackageAffiliation, CatalogPackageRole, CatalogVariantId, ModelPackageId,
};
use serde::Serialize;
use serde_json::Value;

use crate::catalog::{valid_identity_component, valid_variant_id};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CatalogAffiliations {
    entries: BTreeSet<CatalogPackageAffiliation>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StoredCatalogAffiliations<'a> {
    affiliations: Vec<&'a CatalogPackageAffiliation>,
}

impl CatalogAffiliations {
    pub(crate) fn load(root: &Path) -> Self {
        let Ok(bytes) = fs::read(path(root)) else {
            return Self::default();
        };
        let Ok(Value::Object(document)) = serde_json::from_slice::<Value>(&bytes) else {
            tracing::warn!("discarding malformed catalog affiliation cache");
            return Self::default();
        };
        let Some(Value::Array(entries)) = document.get("affiliations") else {
            tracing::warn!("discarding catalog affiliation cache with an invalid shape");
            return Self::default();
        };
        let entries = entries
            .iter()
            .filter_map(decode_entry)
            .collect::<BTreeSet<_>>();
        Self { entries }
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = &CatalogPackageAffiliation> {
        self.entries.iter()
    }

    pub(crate) fn add(&mut self, affiliation: CatalogPackageAffiliation) -> bool {
        if !valid_affiliation(&affiliation) {
            return false;
        }
        self.entries.insert(affiliation)
    }

    pub(crate) fn persist(&self, root: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&StoredCatalogAffiliations {
            affiliations: self.entries.iter().collect(),
        })
        .map_err(std::io::Error::other)?;
        let destination = path(root);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary = root.join(format!("catalog-affiliations.json.tmp-{nonce}"));
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW);
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        if let Err(error) = (|| {
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &destination)?;
            File::open(root)?.sync_all()
        })() {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        Ok(())
    }
}

fn decode_entry(value: &Value) -> Option<CatalogPackageAffiliation> {
    let entry = value.as_object()?;
    let affiliation = CatalogPackageAffiliation {
        model_id: CatalogModelId(entry.get("modelId")?.as_str()?.to_owned()),
        variant_id: CatalogVariantId(entry.get("variantId")?.as_str()?.to_owned()),
        package_id: ModelPackageId(entry.get("packageId")?.as_str()?.to_owned()),
        repository: entry.get("repository")?.as_str()?.to_owned(),
        role: match entry.get("role")?.as_str()? {
            "Target" => CatalogPackageRole::Target,
            "Dependency" => CatalogPackageRole::Dependency,
            _ => return None,
        },
    };
    valid_affiliation(&affiliation).then_some(affiliation)
}

fn valid_affiliation(affiliation: &CatalogPackageAffiliation) -> bool {
    valid_identity_component(&affiliation.model_id.0)
        && valid_variant_id(&affiliation.variant_id.0)
        && !affiliation.package_id.0.is_empty()
        && affiliation.package_id.0.trim() == affiliation.package_id.0
        && valid_repository(&affiliation.repository)
}

fn path(root: &Path) -> std::path::PathBuf {
    root.join("catalog-affiliations.json")
}

fn valid_repository(value: &str) -> bool {
    let Some((owner, name)) = value.split_once('/') else {
        return false;
    };
    valid_repository_component(owner) && valid_repository_component(name) && !name.contains('/')
}

fn valid_repository_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains('\\')
        && !value.contains('\0')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_entries_are_isolated_and_round_trip_deterministically() {
        let root = tempfile::tempdir().expect("temporary store");
        fs::write(
            path(root.path()),
            br#"{
              "unknown": true,
              "affiliations": [
                {"modelId":"model","variantId":"gguf:q4","packageId":"package","repository":"owner/repo","role":"Target"},
                {"modelId":"model","variantId":"gguf:q4","packageId":"package","repository":"owner/repo","role":"Target"},
                {"modelId":"bad:id","variantId":"gguf:q8","packageId":"other","repository":"owner/other","role":"Target"},
                {"modelId":"other","variantId":"invalid","packageId":"other","repository":"owner/other","role":"Dependency"}
              ]
            }"#,
        )
        .expect("affiliations");

        let affiliations = CatalogAffiliations::load(root.path());
        assert_eq!(affiliations.entries.len(), 1);
        affiliations.persist(root.path()).expect("persist");
        assert_eq!(CatalogAffiliations::load(root.path()), affiliations);
    }
}
