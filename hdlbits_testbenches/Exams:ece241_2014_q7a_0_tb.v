module top_module_tb;

    reg clk;
    reg reset;
    reg enable;

    wire [2:0] Q;
    wire c_enable;
    wire c_load;
    wire [2:0] c_d;


    top_module UUT (.clk(clk), .reset(reset), .enable(enable), .Q(Q), .c_enable(c_enable), .c_load(c_load), .c_d(c_d));

    initial // clk generation
    begin
        clk = 0;
        forever begin
            #(period/2);
            clk = ~clk;
        end
    end

    initial begin
        integer mismatch_count;
        mismatch_count = 0;

        // Tick 0: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b001, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b001 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 0: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b001", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 0 passed!");
        end

        // Tick 1: Inputs = High, 4'b0001, 4'b0000, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 8'bx, 4'b0001, 3'b001
        reset = 4'b0001; enable = 4'b0000; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 8'bx && c_load === 4'b0001 && c_d === 3'b001)) begin
            $display("Mismatch at index 1: Inputs = ['High' "4'b0001" "4'b0000"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "8'bx", "4'b0001", "3'b001"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 1 passed!");
        end

        // Tick 2: Inputs = Low, 4'b0001, 4'b0000, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 8'bx, 4'b0001, 3'b001
        reset = 4'b0001; enable = 4'b0000; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 8'bx && c_load === 4'b0001 && c_d === 3'b001)) begin
            $display("Mismatch at index 2: Inputs = ['Low' "4'b0001" "4'b0000"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "8'bx", "4'b0001", "3'b001"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 2 passed!");
        end

        // Tick 3: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b001, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b001 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 3: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b001", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 3 passed!");
        end

        // Tick 4: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b001, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b001 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 4: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b001", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 4 passed!");
        end

        // Tick 5: Inputs = High, 4'b0000, 4'b0000, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 4'b0000, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0000; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 4'b0000 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 5: Inputs = ['High' "4'b0000" "4'b0000"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "4'b0000", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 5 passed!");
        end

        // Tick 6: Inputs = Low, 4'b0000, 4'b0000, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 4'b0000, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0000; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 4'b0000 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 6: Inputs = ['Low' "4'b0000" "4'b0000"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "4'b0000", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 6 passed!");
        end

        // Tick 7: Inputs = High, 4'b0000, 4'b0000, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 4'b0000, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0000; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 4'b0000 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 7: Inputs = ['High' "4'b0000" "4'b0000"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "4'b0000", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 7 passed!");
        end

        // Tick 8: Inputs = Low, 4'b0000, 4'b0000, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 4'b0000, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0000; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 4'b0000 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 8: Inputs = ['Low' "4'b0000" "4'b0000"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "4'b0000", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 8 passed!");
        end

        // Tick 9: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 9: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 9 passed!");
        end

        // Tick 10: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b010, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b010 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 10: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b010", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 10 passed!");
        end

        // Tick 11: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b011, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b011 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 11: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b011", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 11 passed!");
        end

        // Tick 12: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b011, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b011 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 12: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b011", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 12 passed!");
        end

        // Tick 13: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b100, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b100 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 13: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b100", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 13 passed!");
        end

        // Tick 14: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b100, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b100 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 14: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b100", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 14 passed!");
        end

        // Tick 15: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b101, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b101 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 15: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b101", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 15 passed!");
        end

        // Tick 16: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b101, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b101 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 16: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b101", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 16 passed!");
        end

        // Tick 17: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b110, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b110 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 17: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b110", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 17 passed!");
        end

        // Tick 18: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b110, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b110 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 18: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b110", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 18 passed!");
        end

        // Tick 19: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b111, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b111 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 19: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b111", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 19 passed!");
        end

        // Tick 20: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b111, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b111 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 20: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b111", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 20 passed!");
        end

        // Tick 21: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1000, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1000 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 21: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1000", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 21 passed!");
        end

        // Tick 22: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1000, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1000 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 22: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1000", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 22 passed!");
        end

        // Tick 23: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1001, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1001 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 23: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1001", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 23 passed!");
        end

        // Tick 24: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1001, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1001 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 24: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1001", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 24 passed!");
        end

        // Tick 25: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1010, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1010 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 25: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1010", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 25 passed!");
        end

        // Tick 26: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1010, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1010 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 26: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1010", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 26 passed!");
        end

        // Tick 27: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1011, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1011 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 27: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1011", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 27 passed!");
        end

        // Tick 28: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1011, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1011 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 28: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1011", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 28 passed!");
        end

        // Tick 29: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1100, 8'bx, 4'b0001, 3'b001
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1100 && c_enable === 8'bx && c_load === 4'b0001 && c_d === 3'b001)) begin
            $display("Mismatch at index 29: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1100", "8'bx", "4'b0001", "3'b001"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 29 passed!");
        end

        // Tick 30: Inputs = Low, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b1100, 8'bx, 4'b0001, 3'b001
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b1100 && c_enable === 8'bx && c_load === 4'b0001 && c_d === 3'b001)) begin
            $display("Mismatch at index 30: Inputs = ['Low' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b1100", "8'bx", "4'b0001", "3'b001"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 30 passed!");
        end

        // Tick 31: Inputs = High, 4'b0000, 4'b0001, Generated = Q, c_enable, c_load, c_d, Reference = 3'b001, 4'b0001, 4'b0000, 8'bx
        reset = 4'b0000; enable = 4'b0001; // Set input values
        #period;
        if (!(Q === 3'b001 && c_enable === 4'b0001 && c_load === 4'b0000 && c_d === 8'bx)) begin
            $display("Mismatch at index 31: Inputs = ['High' "4'b0000" "4'b0001"], Generated = ['Q', 'c_enable', 'c_load', 'c_d'], Reference = ["3'b001", "4'b0001", "4'b0000", "8'bx"]");
            mismatch_count = mismatch_count + 1;
            $finish;
        end

        else begin
            $display("Test 31 passed!");
        end

        if (mismatch_count == 0)
            $display("All tests passed!");
        else
            $display("%0d mismatches out of %0d total tests.", mismatch_count, 32);
        $finish;
    end

endmodule