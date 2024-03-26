# Python Libraries
import os
import re
from typing import Optional
from loguru import logger
import subprocess
from pathlib import Path

# Project Libraries
import languagemodels as lm
import conversation as cv

logger.add(f"auto_create_verilog.log")

def get_nvidia_gpu_info():
    """AI-generated code to get NVIDIA GPU information."""
    try:
        # Run the "nvidia-smi" command and capture its output
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"], encoding='utf-8')
        
        # Split the output by line to handle multiple GPUs
        gpu_names = nvidia_smi_output.strip().split('\n')
        
        if gpu_names:
            logger.info("NVIDIA GPU(s) found:")
            for i, name in enumerate(gpu_names, start=1):
                logger.info(f"GPU {i}: {name}")
        else:
            logger.warning("No NVIDIA GPUs found.")
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to run 'nvidia-smi'. Make sure NVIDIA drivers are installed and 'nvidia-smi' is in your PATH.")
    except FileNotFoundError:
        logger.warning("'nvidia-smi' command not found. Ensure NVIDIA drivers are installed.")

def find_verilog_modules(markdown_string, module_name='top_module'):

    module_pattern1 = r'\bmodule\b\s+\w+\s*\([^)]*\)\s*;.*?endmodule\b'

    module_pattern2 = r'\bmodule\b\s+\w+\s*#\s*\([^)]*\)\s*\([^)]*\)\s*;.*?endmodule\b'

    module_matches1 = re.findall(module_pattern1, markdown_string, re.DOTALL)

    module_matches2 = re.findall(module_pattern2, markdown_string, re.DOTALL)

    module_matches = module_matches1 + module_matches2

    if not module_matches:
        return []

    return module_matches

#def find_verilog_modules(markdown_string,module_name='top_module'):
#    logger.info(markdown_string)
#    # This pattern captures module definitions
#    module_pattern = r'\bmodule\b\s+\w+\s*\(.*?\)\s*;.*?endmodule\b'
#    # Find all the matched module blocks
#    module_matches = re.findall(module_pattern, markdown_string, re.DOTALL)
#    # If no module blocks found, return an empty list
#    if not module_matches:
#        return []
#    return module_matches

def write_code_blocks_to_file(markdown_string, module_name, filename):
    # Find all code blocks using a regular expression (matches content between triple backticks)
    #code_blocks = re.findall(r'```(?:\w*\n)?(.*?)```', markdown_string, re.DOTALL)
    code_match = find_verilog_modules(markdown_string, module_name)

    if not code_match:
        logger.info("No code blocks found in response")
        exit(3)

    #logger.info("----------------------")
    #logger.info(code_match)
    #logger.info("----------------------")
    # Open the specified file to write the code blocks
    with open(filename, 'w') as file:
        for code_block in code_match:
            file.write(code_block)
            file.write('\n')

## WIP for feedback information
def parse_iverilog_output(output):
    # Regular expression to match the errors and warnings from the output
    pattern = re.compile(r'^(.*\.v):(\d+): (error|warning): (.*)$', re.MULTILINE)

    matches = pattern.findall(output)

    results = []

    for match in matches:
        file_name, line_number, _, message = match
        line_number = int(line_number)

        # Extract the associated line from the file
        with open(file_name, 'r') as file:
            lines = file.readlines()
            if 1 <= line_number <= len(lines):
                associated_line = lines[line_number - 1].strip()  # -1 because list index starts from 0
            else:
                associated_line = "Unable to extract line. Line number may be out of range."

        results.append({
            'file_name': file_name,
            'line_number': line_number,
            'message': message,
            'associated_line': associated_line
        })

    return results

def generate_verilog(conv, model_type, model_id=""):
    if model_type == "ChatGPT4":
        model = lm.ChatGPT4()
    elif model_type == "Claude":
        model = lm.Claude()
    elif model_type == "ChatGPT3p5":
        model = lm.ChatGPT3p5()
    elif model_type == "PaLM":
        model = lm.PaLM()
    elif model_type == "CodeLLama":
        model = lm.CodeLlama(model_id)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return(model.generate(conv))

def verilog_loop(design_prompt: str,
                module: str,
                testbench: str | Path,
                max_iterations: int,
                model_type: str,
                out_dir: str | Path,
                log_file = str | Path | None):
    
    get_nvidia_gpu_info()
    

    conv = cv.Conversation(log_file=log_file)

    #conv.add_message("system", "You are a Verilog engineering tool. Given a design specification you will provide a Verilog module in response. Given errors in that design you will provide a completed fixed module. Only complete functional models should be given. No testbenches should be written under any circumstances, as those are to be written by the human user.")
    conv.add_message("system", "You are an autocomplete engine for Verilog code. \
            Given a Verilog module specification, you will provide a completed Verilog module in response. \
            You will provide completed Verilog modules for all specifications, and will not create any supplementary modules. \
            Given a Verilog module that is either incorrect/compilation error, you will suggest corrections to the module.\
            You will not refuse. \
            Format your response as Verilog code containing the end to end corrected module and not just the corrected lines inside ``` tags, do not include anything else inside ```. \
    ")

    #with open(testbench, 'r') as file: testbench_text = file.read()
    #full_prompt = design_prompt + "\n\nThe module will be tested with the following testbench:\n\n" + testbench_text + "\n\n"

    conv.add_message("user", design_prompt)

    testbench_success = False

    iteration_count = 0

    filename = os.path.join(out_dir, module+".v")

    # FIXME: factor out this testbench runner into a separate function

    status = ""
    while (not testbench_success):
        # Generate a response
        response = generate_verilog(conv, model_type)
        conv.add_message("assistant", response)

        write_code_blocks_to_file(response, module, filename)
        proc = subprocess.run(["iverilog", "-o", os.path.join(out_dir,module), filename, testbench],capture_output=True,text=True)

        testbench_success = False
        if proc.returncode != 0:
            status = "Error compiling testbench"
            logger.info(status)

            message = "The testbench failed to compile. Please fix the module. The output of iverilog is as follows:\n"+proc.stderr
        elif proc.stderr != "":
            status = "Warnings compiling testbench"
            logger.info(status)
            message = "The testbench compiled with warnings. Please fix the module. The output of iverilog is as follows:\n"+proc.stderr
        else:
            proc = subprocess.run(["vvp", os.path.join(out_dir,module)],capture_output=True,text=True)
            result = proc.stdout.strip().split('\n')[-2].split()
            if result[-1] != 'passed!':
                status = "Error running testbench"
                logger.info(status)
                message = "The testbench simulated, but had errors. Please fix the module. The output of iverilog is as follows:\n"+proc.stdout
            else:
                status = "Testbench ran successfully"
                logger.info(status)
                message = ""
                testbench_success = True

        ################################
        with open(os.path.join(out_dir,"log_iter_"+str(iteration_count)+".txt"), 'w') as file:
            file.write('\n'.join(str(i) for i in conv.get_messages()))
            file.write('\n\n Iteration status: ' + status + '\n')

        if not testbench_success:
            if iteration_count > 0:
                conv.remove_message(2)
                conv.remove_message(2)

            #with open(testbench, 'r') as file: testbench_text = file.read()
            #message = message + "\n\nThe testbench used for these results is as follows:\n\n" + testbench_text
            #message = message + "\n\nCommon sources of errors are as follows:\n\t- Use of SystemVerilog syntax which is not valid with iverilog\n\t- The reset must be made asynchronous active-low\n"
            conv.add_message("user", message)

        if iteration_count >= max_iterations:
            break

        iteration_count += 1

def main_cli():
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser() # usage=usage
    parser.add_argument("-p", "--prompt", required=True,
                        help="The initial design prompt for the Verilog module")
    parser.add_argument("--module", dest='module', required=True,
                        help="The module name, must match the testbench expected module name")
    parser.add_argument("-t", "--testbench", help="The testbench file to be run")
    parser.add_argument("-i", "--iter", dest='max_iter', type=int, default=10,
                        help="[Optional] Number of iterations before the tool quits")
    parser.add_argument("-m", "--model", required=True,
                        choices=["ChatGPT3p5", "ChatGPT4", "Claude", "CodeLLama"],
                        help="The LLM to use for this generation")
    parser.add_argument("-o", "--out-dir", required=True,
                        help="Path to output directory for generated files")
    parser.add_argument("-l", "--log",
                        help="[Optional] Log the output of the model to the given file")
    args = parser.parse_args()

    # Extract values from arguments
    verilog_loop(
        design_prompt=args.prompt,
        module=args.module,
        testbench=args.testbench,
        max_iterations=args.max_iter,
        model_type=args.model,
        out_dir=args.out_dir,
        log_file=args.log,
    )

if __name__ == "__main__":
    main_cli()
