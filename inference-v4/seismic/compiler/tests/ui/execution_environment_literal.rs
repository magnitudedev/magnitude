use seismic_compiler::executable::{DeviceService, ExecutionEnvironment};
use seismic_target::TargetFamily;

fn fabricate<'a, T, H, D>() -> ExecutionEnvironment<'a, T, H, D>
where
    T: TargetFamily,
    D: DeviceService<T>,
{
    ExecutionEnvironment {}
}

fn main() {}
