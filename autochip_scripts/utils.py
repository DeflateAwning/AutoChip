import re

def find_verilog_modules(markdown_string, module_name='top_module'):

    module_pattern1 = r'\bmodule\b\s+\w+\s*\([^)]*\)\s*;.*?endmodule\b'

    module_pattern2 = r'\bmodule\b\s+\w+\s*#\s*\([^)]*\)\s*\([^)]*\)\s*;.*?endmodule\b'

    module_matches1 = re.findall(module_pattern1, markdown_string, re.DOTALL)

    module_matches2 = re.findall(module_pattern2, markdown_string, re.DOTALL)

    module_matches = module_matches1 + module_matches2

    if not module_matches:
        return []

    return module_matches
