`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps

module top_module_tb;

    // duration for each bit = 20 * timescale = 20 * 1 ns  = 20ns
    localparam period = 20;

    reg [31:0] a;
    reg [31:0] b;
    reg sub;

    wire [31:0] sum;


    top_module UUT (.a(a), .b(b), .sub(sub), .sum(sum));

    initial begin
        integer mismatch_count;
        mismatch_count = 0;

        // Tick 0: Inputs = 32'b00000000000000000000000000000000, 32'b00000000000000000000000000000000, 4'b0000, Generated = sum, Reference = 32'b00000000000000000000000000000000
        a = 32'b00000000000000000000000000000000; b = 32'b00000000000000000000000000000000; sub = 4'b0000; // Set input values
        #period;
        if (!(sum === 32'b00000000000000000000000000000000)) begin
            $display("Mismatch at index 0: Inputs = ["32'b00000000000000000000000000000000"
 "32'b00000000000000000000000000000000" "4'b0000"], Generated = ['sum'], Reference = ["32'b00000000000000000000000000000000"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 0 passed!");
        end

        // Tick 1: Inputs = 32'b00000000000000000000000000000001, 32'b00000000000000000000000000000000, 4'b0000, Generated = sum, Reference = 32'b00000000000000000000000000000001
        a = 32'b00000000000000000000000000000001; b = 32'b00000000000000000000000000000000; sub = 4'b0000; // Set input values
        #period;
        if (!(sum === 32'b00000000000000000000000000000001)) begin
            $display("Mismatch at index 1: Inputs = ["32'b00000000000000000000000000000001"
 "32'b00000000000000000000000000000000" "4'b0000"], Generated = ['sum'], Reference = ["32'b00000000000000000000000000000001"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 1 passed!");
        end

        // Tick 2: Inputs = 32'b00000000000000000000000000000010, 32'b00000000000000000000000000000000, 4'b0000, Generated = sum, Reference = 32'b00000000000000000000000000000010
        a = 32'b00000000000000000000000000000010; b = 32'b00000000000000000000000000000000; sub = 4'b0000; // Set input values
        #period;
        if (!(sum === 32'b00000000000000000000000000000010)) begin
            $display("Mismatch at index 2: Inputs = ["32'b00000000000000000000000000000010"
 "32'b00000000000000000000000000000000" "4'b0000"], Generated = ['sum'], Reference = ["32'b00000000000000000000000000000010"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 2 passed!");
        end

        // Tick 3: Inputs = 32'b00000000000000000000000000000010, 32'b00000000000000000000000000000001, 4'b0001, Generated = sum, Reference = 32'b00000000000000000000000000000001
        a = 32'b00000000000000000000000000000010; b = 32'b00000000000000000000000000000001; sub = 4'b0001; // Set input values
        #period;
        if (!(sum === 32'b00000000000000000000000000000001)) begin
            $display("Mismatch at index 3: Inputs = ["32'b00000000000000000000000000000010"
 "32'b00000000000000000000000000000001" "4'b0001"], Generated = ['sum'], Reference = ["32'b00000000000000000000000000000001"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 3 passed!");
        end

        // Tick 4: Inputs = 32'b00000000000000000000000000000010, 32'b00000000000000000000000000000010, 4'b0001, Generated = sum, Reference = 32'b00000000000000000000000000000000
        a = 32'b00000000000000000000000000000010; b = 32'b00000000000000000000000000000010; sub = 4'b0001; // Set input values
        #period;
        if (!(sum === 32'b00000000000000000000000000000000)) begin
            $display("Mismatch at index 4: Inputs = ["32'b00000000000000000000000000000010"
 "32'b00000000000000000000000000000010" "4'b0001"], Generated = ['sum'], Reference = ["32'b00000000000000000000000000000000"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 4 passed!");
        end

        // Tick 5: Inputs = 32'b00000000000000001111111111111111, 32'b00000000000000000000000000000001, 4'b0000, Generated = sum, Reference = 32'b00000000000000010000000000000000
        a = 32'b00000000000000001111111111111111; b = 32'b00000000000000000000000000000001; sub = 4'b0000; // Set input values
        #period;
        if (!(sum === 32'b00000000000000010000000000000000)) begin
            $display("Mismatch at index 5: Inputs = ["32'b00000000000000001111111111111111"
 "32'b00000000000000000000000000000001" "4'b0000"], Generated = ['sum'], Reference = ["32'b00000000000000010000000000000000"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 5 passed!");
        end

        // Tick 6: Inputs = 32'b00000000000000010000000000000000, 32'b00000000000000000000000000000001, 4'b0001, Generated = sum, Reference = 32'b00000000000000001111111111111111
        a = 32'b00000000000000010000000000000000; b = 32'b00000000000000000000000000000001; sub = 4'b0001; // Set input values
        #period;
        if (!(sum === 32'b00000000000000001111111111111111)) begin
            $display("Mismatch at index 6: Inputs = ["32'b00000000000000010000000000000000"
 "32'b00000000000000000000000000000001" "4'b0001"], Generated = ['sum'], Reference = ["32'b00000000000000001111111111111111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 6 passed!");
        end

        // Tick 7: Inputs = 32'b11111111111111111111111111111111, 32'b11111111111111111111111111111111, 4'b0000, Generated = sum, Reference = 32'b11111111111111111111111111111110
        a = 32'b11111111111111111111111111111111; b = 32'b11111111111111111111111111111111; sub = 4'b0000; // Set input values
        #period;
        if (!(sum === 32'b11111111111111111111111111111110)) begin
            $display("Mismatch at index 7: Inputs = ["32'b11111111111111111111111111111111"
 "32'b11111111111111111111111111111111" "4'b0000"], Generated = ['sum'], Reference = ["32'b11111111111111111111111111111110"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 7 passed!");
        end

        if (mismatch_count == 0)
            $display("All tests passed!");
        else
            $display("%0d mismatches out of %0d total tests.", mismatch_count, 8);
        $finish;
    end

endmodule