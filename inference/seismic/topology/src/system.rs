//! The machine itself, as its firmware names it: its maker, model, family
//! and version, its enclosure, and the name people call it by. Never a
//! serial number, UUID or asset tag: no reader looks at one.
//!
//! Firmware may name nothing (a VM, a container, a board without tables),
//! or write placeholders a vendor never filled in; neither is a name, so
//! either reads as absent rather than as an error.

/// The system identity as firmware reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SystemIdentity {
    /// The maker and model, where the firmware names both.
    pub identity: Option<HardwareIdentity>,
    pub form_factor: FormFactor,
    /// What people call the machine: macOS's marketing name ("MacBook
    /// Pro"), and elsewhere its model ("XPS 15 9530"), or a board's own
    /// name from its device tree where it has no firmware tables.
    pub name: Option<String>,
}

/// The firmware's names for the machine: SMBIOS's system information
/// (type 1), or on macOS Apple and `hw.model`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareIdentity {
    pub manufacturer: String,
    pub model: String,
    pub family: Option<String>,
    pub version: Option<String>,
}

/// The enclosure, from SMBIOS's chassis type (type 3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FormFactor {
    Portable,
    Desktop,
    AllInOne,
    MiniPc,
    Server,
    #[default]
    Unknown,
}

/// Raw firmware strings before placeholder filtering.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Firmware {
    pub manufacturer: Vec<u8>,
    pub model: Vec<u8>,
    pub family: Vec<u8>,
    pub version: Vec<u8>,
    pub chassis_type: Option<u8>,
    /// The machine's name, where the host names it apart from its model.
    pub name: Vec<u8>,
}

impl SystemIdentity {
    /// Normalize raw firmware fields into a [`SystemIdentity`].
    pub fn from_firmware(firmware: Firmware) -> Self {
        let model = label(&firmware.model);
        let identity = label(&firmware.manufacturer)
            .zip(model.clone())
            .map(|(manufacturer, model)| HardwareIdentity {
                manufacturer,
                model,
                family: label(&firmware.family),
                version: label(&firmware.version),
            });
        Self {
            identity,
            form_factor: firmware
                .chassis_type
                .map_or(FormFactor::Unknown, form_factor),
            // A model's underscores ("NVIDIA_DGX_Spark") are its firmware's spelling, not its name's.
            name: label(&firmware.name)
                .or(model)
                .map(|name| name.replace('_', " ")),
        }
    }
}

/// `raw` as a name: control characters and runs of space as one space,
/// trimmed; none where nothing is left, where it runs past 255 bytes, or
/// where it is a placeholder.
pub fn label(raw: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(raw);
    let name = text
        .split(|character: char| character.is_control() || character.is_whitespace())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let placeholder = [
        "unknown",
        "none",
        "not specified",
        "not applicable",
        "default string",
        "system manufacturer",
        "system product name",
        "system version",
        "system family",
        "to be filled by o.e.m.",
        "to be filled by oem",
    ]
    .iter()
    .any(|placeholder| name.eq_ignore_ascii_case(placeholder));
    (!name.is_empty() && name.len() <= 255 && !placeholder).then_some(name)
}

/// SMBIOS's chassis type, its lock bit masked.
fn form_factor(chassis_type: u8) -> FormFactor {
    match chassis_type & 0x7f {
        8 | 9 | 10 | 11 | 14 | 30 | 31 | 32 => FormFactor::Portable,
        3 | 4 | 5 | 6 | 7 | 15 | 16 => FormFactor::Desktop,
        13 => FormFactor::AllInOne,
        35 | 36 => FormFactor::MiniPc,
        17 | 23 | 28 | 29 => FormFactor::Server,
        _ => FormFactor::Unknown,
    }
}

/// The SMBIOS structure table `table` (without Windows' eight-byte header):
/// the first system information structure's names, and the first chassis
/// structure's type, in either order. A structure that runs past the table
/// ends the walk; a string index past a structure's strings, or a string
/// past 255 bytes, names nothing.
pub fn parse_smbios(table: &[u8]) -> Firmware {
    let mut firmware = Firmware::default();
    let (mut system, mut chassis) = (false, false);
    let mut rest = table;
    while rest.len() >= 4 {
        let (kind, formatted) = (rest[0], usize::from(rest[1]));
        if formatted < 4 || formatted > rest.len() || kind == 127 {
            break;
        }
        // The strings end at the first double NUL after the formatted area.
        let Some(end) = rest[formatted..]
            .windows(2)
            .position(|pair| pair == [0, 0])
            .map(|at| formatted + at)
        else {
            break;
        };
        let (fields, strings) = (&rest[..formatted], &rest[formatted..end + 1]);
        let string = |index: u8| -> Vec<u8> {
            let found = (index > 0)
                .then(|| strings.split(|byte| *byte == 0).nth(usize::from(index) - 1))
                .flatten()
                .filter(|text| !text.is_empty() && text.len() <= 255);
            found.map(<[u8]>::to_vec).unwrap_or_default()
        };
        if kind == 1 && formatted >= 6 && !system {
            system = true;
            firmware.manufacturer = string(fields[4]);
            firmware.model = string(fields[5]);
            if formatted >= 7 {
                firmware.version = string(fields[6]);
            }
            if formatted >= 27 {
                firmware.family = string(fields[26]);
            }
        }
        if kind == 3 && formatted >= 6 && !chassis {
            chassis = true;
            firmware.chassis_type = Some(fields[5]);
        }
        rest = &rest[end + 2..];
    }
    firmware
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records() -> (Vec<u8>, Vec<u8>) {
        let mut system = vec![1, 27, 0, 0, 1, 2, 3, 5];
        system.resize(26, 0);
        system.push(4);
        system.extend_from_slice(b"Lenovo\x0021HD\x00T14\x00ThinkPad\x00SECRET\x00\x00");
        (system, vec![3, 6, 0, 0, 0, 0x8a, 0, 0])
    }

    #[test]
    fn smbios_names_the_system_and_its_enclosure_in_either_order() {
        let (system, chassis) = records();
        for table in [[system.clone(), chassis.clone()].concat(), [chassis, system].concat()] {
            let machine = SystemIdentity::from_firmware(parse_smbios(&table));
            assert_eq!(
                machine.identity,
                Some(HardwareIdentity {
                    manufacturer: "Lenovo".into(),
                    model: "21HD".into(),
                    family: Some("ThinkPad".into()),
                    version: Some("T14".into()),
                })
            );
            assert_eq!(machine.form_factor, FormFactor::Portable);
            assert_eq!(machine.name.as_deref(), Some("21HD"));
        }
    }

    #[test]
    fn smbios_reads_nothing_from_a_truncated_or_malformed_table() {
        let (system, _) = records();
        for length in 0..system.len() {
            assert!(parse_smbios(&system[..length]).model.is_empty());
        }
        let mut indexed = system.clone();
        indexed[4] = 99;
        indexed[5] = 0;
        assert!(parse_smbios(&indexed).manufacturer.is_empty() && parse_smbios(&indexed).model.is_empty());
        let mut overlong = vec![1, 6, 0, 0, 1, 1];
        overlong.extend(std::iter::repeat_n(b'X', 300));
        overlong.extend_from_slice(&[0, 0]);
        assert!(parse_smbios(&overlong).model.is_empty());
        let ended = [vec![127, 4, 0, 0], system].concat();
        assert!(parse_smbios(&ended).model.is_empty());
        let mut random = 1_234_567u32;
        for round in 0..2_000 {
            let table = (0..round % 1_024)
                .map(|_| {
                    random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (random >> 24) as u8
                })
                .collect::<Vec<_>>();
            parse_smbios(&table);
        }
    }

    #[test]
    fn placeholders_and_unusable_labels_are_no_name() {
        for model in [
            &b""[..],
            b"System Product Name",
            b"Default string",
            b"To Be Filled By O.E.M.",
            b"Not Applicable",
            &[b'x'; 256],
        ] {
            let machine = SystemIdentity::from_firmware(Firmware {
                manufacturer: b"Vendor".to_vec(),
                model: model.to_vec(),
                chassis_type: Some(10),
                ..Firmware::default()
            });
            assert_eq!((machine.identity, machine.name, machine.form_factor), (None, None, FormFactor::Portable));
        }
        let machine = SystemIdentity::from_firmware(Firmware {
            manufacturer: b" Lenovo\n".to_vec(),
            model: b"21HD\0".to_vec(),
            family: b"System Family".to_vec(),
            version: b"Default string".to_vec(),
            ..Firmware::default()
        });
        let identity = machine.identity.unwrap();
        assert_eq!((identity.manufacturer.as_str(), identity.model.as_str()), ("Lenovo", "21HD"));
        assert_eq!((identity.family, identity.version), (None, None));
    }

    #[test]
    fn a_name_is_the_hosts_own_or_the_models_spelled_as_a_name() {
        let board = SystemIdentity::from_firmware(Firmware {
            name: b"Raspberry Pi 5 Model B Rev 1.0\0".to_vec(),
            ..Firmware::default()
        });
        assert_eq!((board.identity, board.name.as_deref()), (None, Some("Raspberry Pi 5 Model B Rev 1.0")));
        let spark = SystemIdentity::from_firmware(Firmware {
            manufacturer: b"NVIDIA".to_vec(),
            model: b"NVIDIA_DGX_Spark".to_vec(),
            ..Firmware::default()
        });
        assert_eq!(spark.identity.unwrap().model, "NVIDIA_DGX_Spark");
        assert_eq!(spark.name.as_deref(), Some("NVIDIA DGX Spark"));
    }

    #[test]
    fn chassis_types_classify_their_enclosure() {
        for (kind, expected) in [
            (8, FormFactor::Portable),
            (10, FormFactor::Portable),
            (32, FormFactor::Portable),
            (3, FormFactor::Desktop),
            (13, FormFactor::AllInOne),
            (35, FormFactor::MiniPc),
            (23, FormFactor::Server),
            (0x8a, FormFactor::Portable),
            (0, FormFactor::Unknown),
            (255, FormFactor::Unknown),
        ] {
            assert_eq!(form_factor(kind), expected, "{kind}");
        }
    }
}
