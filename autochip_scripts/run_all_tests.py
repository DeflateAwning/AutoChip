# Python Libraries
from loguru import logger
from pathlib import Path
from datetime import datetime

# Project Imports
from auto_create_verilog import verilog_loop

def main(model_type: str, tests_per_prompt: int = 5):
    repo_root = Path(__file__).parent.parent
    
    prompt_dir = repo_root / 'hdlbits_prompts'
    testbench_dir = repo_root / 'hdlbits_testbenches'
    output_dir = repo_root / 'outputs' / f'outputs_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    logger.info(f"Output directory: {output_dir}")

    # Collect all prompt names, without their extensions
    prompt_file_paths: list[Path] = list(prompt_dir.glob('*.v'))

    logger.info(f"Collected {len(prompt_file_paths)} prompts (e.g., {prompt_file_paths[:5]})")

    logger.info(f"Beginning main loop!")

    # Main loop
    for prompt_file_path in prompt_file_paths:
        testbench_path = testbench_dir / f"{prompt_file_path.stem}_0_tb.v"

        # TODO: add support for _1_tb.v, _2_tb.v, etc.
        
        # Check if the matching testbench exists
        if not testbench_path.is_file():
            logger.info(f"No matching testbench for {prompt_file_path.name}. Skipping.")
            continue

        for test_num in range(tests_per_prompt):
            test_output_dir = output_dir / prompt_file_path.stem / f"test_{test_num}"
            test_output_dir.mkdir(parents=True, exist_ok=True)

            with open(prompt_file_path, 'r') as f:
                prompt_contents = f.read()

            logger.info(f"Running test {test_num} for {prompt_file_path.name}")

            verilog_loop(
                design_prompt=prompt_contents,
                module="top_module", # FIXME: this is a dumb arg
                testbench=testbench_path,
                max_iterations=tests_per_prompt,
                model_type=model_type,
                out_dir=test_output_dir,
                conversation_log_file=(test_output_dir / "log.txt"),
            )

if __name__ == '__main__':
    main(
        model_type="CodeLLama",
        tests_per_prompt=5,
    )
