#!/bin/bash

# Run this script from the <repo_root>/autochip_scripts directory!

# halt if there's an error
set -e

prompt_dir=`pwd`/hdlbits_prompts
testbench_dir=`pwd`/hdlbits_testbenches
output_dir=`pwd`/outputs

autogen_script=`pwd`/autochip_scripts/auto_create_verilog.py

source `pwd`/venv/bin/activate

echo "Source complete"

tests_per_prompt=5

promtps=()

for path in "$prompt_dir"/*; do
	if [[ -f "$path" ]]; then
		prompt_name=$(basename "$path")
		prompts+=("${prompt_name%.*}")
	fi
done

echo "Prompts: ${prompts[@]}"

echo "Starting loop per-prompt"
date

for prompt in "${prompts[@]}"; do
	#check if there's a matching testbench
	testbench="${testbench_dir}/${prompt}_0_tb.v"
	if [[ ! -f $testbench ]]; then
		echo "No matching testbench for ${prompt}"
		continue
	fi

	for ((i=0; i<tests_per_prompt; i++)); do
		mkdir -p $output_dir/$prompt/test_${i}
		cd $output_dir/$prompt/test_${i}

		python3 $autogen_script --prompt="$(cat $prompt_dir/${prompt}.v)" --testbench=$testbench --module=top_module --iter=10 --model=ChatGPT3p5 --log=${prompt}_log.txt
		cd -
	done
done

echo "Done"
date
