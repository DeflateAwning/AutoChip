`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps

module top_module_tb;

    // duration for each bit = 20 * timescale = 20 * 1 ns  = 20ns
    localparam period = 20;

    reg [99:0] a;
    reg b;
    reg sel;

    wire [99:0] out;


    top_module UUT (.a(a), .b(b), .sel(sel), .out(out));

    initial begin
        integer mismatch_count;
        mismatch_count = 0;

        // Tick 0: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0000, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0000; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111)) begin
            $display("Mismatch at index 0: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0000"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 0 passed!");
        end

        // Tick 1: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0001, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0001; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101)) begin
            $display("Mismatch at index 1: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0001"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 1 passed!");
        end

        // Tick 2: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0001, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0001; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101)) begin
            $display("Mismatch at index 2: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0001"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 2 passed!");
        end

        // Tick 3: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0000, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0000; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111)) begin
            $display("Mismatch at index 3: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0000"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 3 passed!");
        end

        // Tick 4: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0000, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0000; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111)) begin
            $display("Mismatch at index 4: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0000"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 4 passed!");
        end

        // Tick 5: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0001, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0001; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101)) begin
            $display("Mismatch at index 5: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0001"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 5 passed!");
        end

        // Tick 6: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0001, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0001; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101)) begin
            $display("Mismatch at index 6: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0001"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 6 passed!");
        end

        // Tick 7: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0000, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0000; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111)) begin
            $display("Mismatch at index 7: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0000"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 7 passed!");
        end

        // Tick 8: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0000, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0000; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111)) begin
            $display("Mismatch at index 8: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0000"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 8 passed!");
        end

        // Tick 9: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0001, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0001; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101)) begin
            $display("Mismatch at index 9: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0001"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 9 passed!");
        end

        // Tick 10: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0001, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0001; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101)) begin
            $display("Mismatch at index 10: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0001"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000000000101111010101111000000001101"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 10 passed!");
        end

        // Tick 11: Inputs = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111, 28'b0101111010101111000000001101, 4'b0000, Generated = out, Reference = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111
        a = 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111; b = 28'b0101111010101111000000001101; sel = 4'b0000; // Set input values
        #period;
        if (!(out === 100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111)) begin
            $display("Mismatch at index 11: Inputs = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"
 "28'b0101111010101111000000001101" "4'b0000"], Generated = ['out'], Reference = ["100'b0000000000000000000000000000000000000000000000000000000000000000000011011110101011011011111011101111"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 11 passed!");
        end

        if (mismatch_count == 0)
            $display("All tests passed!");
        else
            $display("%0d mismatches out of %0d total tests.", mismatch_count, 12);
        $finish;
    end

endmodule