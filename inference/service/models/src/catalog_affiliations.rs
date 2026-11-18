use std::collections::BTreeSet;
use std::path::Path;

use magnitude_service_contracts::models::{
    CatalogBaseId, CatalogPackageAffiliation, CatalogPackageRole, CatalogVariantId, ModelPackageId,
};
use serde::Serialize;
use serde_json::Value;

use crate::catalog::{valid_identity_component, valid_variant_id};

/// Which installed packages are which catalog models', kept in the store.
///
/// Every service process over one store holds them in memory, and each persist applies what this
/// process changed to what the file holds then, under the file's record lock
/// (`seismic_io::Store::lock`), and takes what that leaves: no process drops another's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CatalogAffiliations {
    entries: BTreeSet<CatalogPackageAffiliation>,
    pending: Pending,
}

/// What this process changed since its last persist: the packages it forgot, applied first, and
/// the affiliations it added.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Pending {
    forgotten: BTreeSet<ModelPackageId>,
    added: BTreeSet<CatalogPackageAffiliation>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StoredCatalogAffiliations<'a> {
    affiliations: Vec<&'a CatalogPackageAffiliation>,
}

const FILE_NAME: &str = "catalog-affiliations.json";

/// The most bytes the file is read as.
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// The affiliations a file's `bytes` hold; an entry that does not read is dropped, a file that
/// does not is none.
fn decode(bytes: &[u8]) -> BTreeSet<CatalogPackageAffiliation> {
    let Ok(Value::Object(document)) = serde_json::from_slice::<Value>(bytes) else {
        tracing::warn!("discarding malformed catalog affiliation cache");
        return BTreeSet::new();
    };
    let Some(Value::Array(entries)) = document.get("affiliations") else {
        tracing::warn!("discarding catalog affiliation cache with an invalid shape");
        return BTreeSet::new();
    };
    entries.iter().filter_map(decode_entry).collect()
}

impl CatalogAffiliations {
    pub(crate) fn load(root: &Path) -> Self {
        let entries = match crate::store_fs::model_store(root).read(Path::new(FILE_NAME), MAX_FILE_BYTES) {
            Ok(Some(bytes)) => decode(&bytes),
            Ok(None) => BTreeSet::new(),
            Err(error) => {
                tracing::warn!(%error, "discarding an unreadable catalog affiliation cache");
                BTreeSet::new()
            }
        };
        Self {
            entries,
            pending: Pending::default(),
        }
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = &CatalogPackageAffiliation> {
        self.entries.iter()
    }

    pub(crate) fn add(&mut self, affiliation: CatalogPackageAffiliation) -> bool {
        if !valid_affiliation(&affiliation) || !self.entries.insert(affiliation.clone()) {
            return false;
        }
        self.pending.added.insert(affiliation);
        true
    }

    /// Forgets every affiliation of `package_id`; whether there was one.
    pub(crate) fn forget_package(&mut self, package_id: &ModelPackageId) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|affiliation| &affiliation.package_id != package_id);
        if self.entries.len() == before {
            return false;
        }
        self.pending
            .added
            .retain(|affiliation| &affiliation.package_id != package_id);
        self.pending.forgotten.insert(package_id.clone());
        true
    }

    /// Apply what this process changed since its last persist to what the file holds, under the
    /// file's record lock, and hold what that leaves, every other process's changes among it. A
    /// failure keeps the change for the next persist.
    pub(crate) fn persist(&mut self, root: &Path) -> std::io::Result<()> {
        if self.pending == Pending::default() {
            return Ok(());
        }
        let store = crate::store_fs::model_store(root);
        let record = store.lock(Path::new(FILE_NAME))?;
        let mut entries = record.read(MAX_FILE_BYTES)?.map(|bytes| decode(&bytes)).unwrap_or_default();
        entries.retain(|affiliation| !self.pending.forgotten.contains(&affiliation.package_id));
        entries.extend(self.pending.added.iter().cloned());
        let bytes = serde_json::to_vec_pretty(&StoredCatalogAffiliations {
            affiliations: entries.iter().collect(),
        })
        .map_err(std::io::Error::other)?;
        record.write(&bytes, seismic_io::Durability::Durable)?;
        self.entries = entries;
        self.pending = Pending::default();
        Ok(())
    }
}

fn decode_entry(value: &Value) -> Option<CatalogPackageAffiliation> {
    let entry = value.as_object()?;
    let affiliation = CatalogPackageAffiliation {
        model_id: CatalogBaseId::new(entry.get("modelId")?.as_str()?.to_owned()).ok()?,
        variant_id: CatalogVariantId::new(entry.get("variantId")?.as_str()?.to_owned()).ok()?,
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
    valid_identity_component(affiliation.model_id.as_str())
        && valid_variant_id(affiliation.variant_id.as_str())
        && !affiliation.package_id.0.is_empty()
        && affiliation.package_id.0.trim() == affiliation.package_id.0
        && valid_repository(&affiliation.repository)
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
    use std::fs;

    fn path(root: &Path) -> std::path::PathBuf {
        root.join(FILE_NAME)
    }

    fn affiliation(package: &str) -> CatalogPackageAffiliation {
        CatalogPackageAffiliation {
            model_id: CatalogBaseId::new("model".to_owned()).unwrap(),
            variant_id: CatalogVariantId::new("gguf:q4".to_owned()).unwrap(),
            package_id: ModelPackageId(package.to_owned()),
            repository: "owner/repo".to_owned(),
            role: CatalogPackageRole::Target,
        }
    }

    /// Two service processes over one store, each adding and forgetting packages of its own: the
    /// file holds what both changed, and each then holds what the other added.
    #[test]
    fn two_processes_keep_each_others_affiliations() {
        let root = tempfile::tempdir().expect("temporary store");
        let mut first = CatalogAffiliations::load(root.path());
        let mut second = CatalogAffiliations::load(root.path());
        assert!(first.add(affiliation("kept")));
        assert!(first.add(affiliation("forgotten")));
        first.persist(root.path()).expect("persist");
        assert!(second.add(affiliation("theirs")));
        second.persist(root.path()).expect("persist");
        assert!(first.forget_package(&ModelPackageId("forgotten".to_owned())));
        first.persist(root.path()).expect("persist");
        let held = |affiliations: &CatalogAffiliations| {
            affiliations.entries().map(|affiliation| affiliation.package_id.0.clone()).collect::<Vec<_>>()
        };
        assert_eq!(held(&CatalogAffiliations::load(root.path())), ["kept", "theirs"]);
        assert_eq!(held(&first), ["kept", "theirs"], "a persist takes what the others changed");
    }

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

        let mut affiliations = CatalogAffiliations::load(root.path());
        assert_eq!(affiliations.entries.len(), 1);
        // A persist writes only a change: one rewrites the file from what it held.
        assert!(affiliations.add(affiliation("added")));
        affiliations.persist(root.path()).expect("persist");
        assert_eq!(affiliations.entries.len(), 2);
        assert_eq!(CatalogAffiliations::load(root.path()), affiliations);
    }
}
