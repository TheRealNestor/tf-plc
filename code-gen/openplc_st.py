from .st_code import STCode

def generate_program_wrapper(
    fb_name: str, program_name: str = "prog0", instance_name: str = "nn"
) -> STCode:
    """Generate a PROGRAM wrapper that instantiates and calls the function block."""
    return STCode.from_lines(
        f"PROGRAM {program_name}",
        "VAR",
        f"    {instance_name} : {fb_name};",
        "END_VAR",
        "",
        f"{instance_name}();",
        "",
        "END_PROGRAM",
        "",
    )


def generate_openplc_configuration(
    program_name: str = "prog0",
    configuration_name: str = "Config0",
    resource_name: str = "Res0",
    task_name: str = "Main",
    task_interval: str = "T#1000ms",
    task_priority: int = 0,
    instance_name: str = "Inst0",
) -> STCode:
    """Generate OpenPLC configuration footer (CONFIGURATION / RESOURCE / TASK mapping)."""
    return STCode.from_lines(
        f"CONFIGURATION {configuration_name}",
        "",
        f"  RESOURCE {resource_name} ON PLC",
        f"    TASK {task_name}(INTERVAL := {task_interval},PRIORITY := {task_priority});",
        f"    PROGRAM {instance_name} WITH {task_name} : {program_name};",
        "  END_RESOURCE",
        "END_CONFIGURATION",
        "",
    )
