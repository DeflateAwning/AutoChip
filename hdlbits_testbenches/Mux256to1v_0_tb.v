`timescale 1 ns/10 ps  // time-unit = 1 ns, precision = 10 ps

module top_module_tb;

    // duration for each bit = 20 * timescale = 20 * 1 ns  = 20ns
    localparam period = 20;

    reg [1023:0] in;
    reg [7:0] sel;

    wire [3:0] out;


    integer mismatch_count;

    top_module UUT (.in(in), .sel(sel), .out(out));

    initial begin
        mismatch_count = 0;

        // Tick 0: Inputs = 1024'b0000010101110011100001110000101010110001111011110110001001100011101100101010011100100110011001011001011010101011010110000010110111011110100011100010100010111101001011100101100001001001010111001110001011001010010011101100010111110100000000000111101011101000111001110111011010010110110011100111100100110000011010011111001001000111111011001101101110001111100010010011001011010110000100101011101111010010011100100111011101110010101011111111011111100101110101010001001111010010101010101110001011110111100001001100010111100011001101110010010011000110011111001111110111101001111110010100011000101101111101111000110001110110110101000101011111101101000111101000110111001101001111010011101100100011111100010111011000000110110101111100110100001101000000001111001111100011000000011000100100110111010100100001001010110010110000101000010001100101010001101101111110011001100011010000011010111001011110110000110110110001111100000101011001100011100001001000010011010110000010011100000010001001010111101000000100010010000101010011010100100100, 8'b10000000, Generated = out, Reference = 4'b0101
        in = 1024'b0000010101110011100001110000101010110001111011110110001001100011101100101010011100100110011001011001011010101011010110000010110111011110100011100010100010111101001011100101100001001001010111001110001011001010010011101100010111110100000000000111101011101000111001110111011010010110110011100111100100110000011010011111001001000111111011001101101110001111100010010011001011010110000100101011101111010010011100100111011101110010101011111111011111100101110101010001001111010010101010101110001011110111100001001100010111100011001101110010010011000110011111001111110111101001111110010100011000101101111101111000110001110110110101000101011111101101000111101000110111001101001111010011101100100011111100010111011000000110110101111100110100001101000000001111001111100011000000011000100100110111010100100001001010110010110000101000010001100101010001101101111110011001100011010000011010111001011110110000110110110001111100000101011001100011100001001000010011010110000010011100000010001001010111101000000100010010000101010011010100100100; sel = 8'b10000000; // Set input values
        #period;
        if (!(out === 4'b0101)) begin
            $display("Mismatch at index 0: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b0000010101110011100001110000101010110001111011110110001001100011101100101010011100100110011001011001011010101011010110000010110111011110100011100010100010111101001011100101100001001001010111001110001011001010010011101100010111110100000000000111101011101000111001110111011010010110110011100111100100110000011010011111001001000111111011001101101110001111100010010011001011010110000100101011101111010010011100100111011101110010101011111111011111100101110101010001001111010010101010101110001011110111100001001100010111100011001101110010010011000110011111001111110111101001111110010100011000101101111101111000110001110110110101000101011111101101000111101000110111001101001111010011101100100011111100010111011000000110110101111100110100001101000000001111001111100011000000011000100100110111010100100001001010110010110000101000010001100101010001101101111110011001100011010000011010111001011110110000110110110001111100000101011001100011100001001000010011010110000010011100000010001001010111101000000100010010000101010011010100100100, 8'b10000000, out, 4'b0101);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 0 passed!");
        end

        // Tick 1: Inputs = 1024'b0100001011110010010000011000010110111000100101111011111001110001100001011101011110011010000010110001010100001111110111010010101011011110011101010000001010111100010101110001010100010011101011100110001101001011111110011100011001011011000000100110010110110110011101011100010100001101111010111100010010001010000100101000100100111100001000001111001101111000111011000100101100110100110110000010000011000100101100110100000101000101001011100110000110001010011110010110100010111101111100101001111000110001010011000011110011100101011100110000101011001010000001010000100101100101000010100001000110000100010010010010001111100111110001010111001011001111000011101111111111101001000111011101011101010110001111101010111010000001000101110100101000000010111010101010011000101010110101010011010110011111110111010110101110101001101001111101011001010011100001101011110000111000000011011000100110000011101110000001001111001011001000000011111010010110110011101100110011001100100111010101010101111000010001011010101000010000011001000010000100100000, 8'b01001111, Generated = out, Reference = 4'b1000
        in = 1024'b0100001011110010010000011000010110111000100101111011111001110001100001011101011110011010000010110001010100001111110111010010101011011110011101010000001010111100010101110001010100010011101011100110001101001011111110011100011001011011000000100110010110110110011101011100010100001101111010111100010010001010000100101000100100111100001000001111001101111000111011000100101100110100110110000010000011000100101100110100000101000101001011100110000110001010011110010110100010111101111100101001111000110001010011000011110011100101011100110000101011001010000001010000100101100101000010100001000110000100010010010010001111100111110001010111001011001111000011101111111111101001000111011101011101010110001111101010111010000001000101110100101000000010111010101010011000101010110101010011010110011111110111010110101110101001101001111101011001010011100001101011110000111000000011011000100110000011101110000001001111001011001000000011111010010110110011101100110011001100100111010101010101111000010001011010101000010000011001000010000100100000; sel = 8'b01001111; // Set input values
        #period;
        if (!(out === 4'b1000)) begin
            $display("Mismatch at index 1: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b0100001011110010010000011000010110111000100101111011111001110001100001011101011110011010000010110001010100001111110111010010101011011110011101010000001010111100010101110001010100010011101011100110001101001011111110011100011001011011000000100110010110110110011101011100010100001101111010111100010010001010000100101000100100111100001000001111001101111000111011000100101100110100110110000010000011000100101100110100000101000101001011100110000110001010011110010110100010111101111100101001111000110001010011000011110011100101011100110000101011001010000001010000100101100101000010100001000110000100010010010010001111100111110001010111001011001111000011101111111111101001000111011101011101010110001111101010111010000001000101110100101000000010111010101010011000101010110101010011010110011111110111010110101110101001101001111101011001010011100001101011110000111000000011011000100110000011101110000001001111001011001000000011111010010110110011101100110011001100100111010101010101111000010001011010101000010000011001000010000100100000, 8'b01001111, out, 4'b1000);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 1 passed!");
        end

        // Tick 2: Inputs = 1024'b1110100111101011111101101101001100001111110100101000111100011111000111001101100111100111001110010011011011100101100000010110110110111111000001010000000001111110000010010000110011011011000100100001111011110010111011010011110110111011100000100101101001110111011001000101011111101101110010001110000100101100110011101100001000000110000111010111111100001100010010110010000100101111100101101010100011000111111111000101000111101011111111101100000011010111111010000010001100111110110100001010010010101110001100100100100101000100110111100011011110001001101011011100101111000000010110111010111001111101100101000101110011001111110001000101011010011111110110111100110101100000101101110111110001101101101010011111100001000111101110011010000110001111010011111010000101010101100111110010011000110101111110110100110000110001001000110000011101100010011011001001110001001011110110010111100011011001100110111111000100001010101010100100101100010101101111110010001100110010011111100001110100000110001100110011101010011101110011000110000000111011, 8'b10000101, Generated = out, Reference = 4'b1010
        in = 1024'b1110100111101011111101101101001100001111110100101000111100011111000111001101100111100111001110010011011011100101100000010110110110111111000001010000000001111110000010010000110011011011000100100001111011110010111011010011110110111011100000100101101001110111011001000101011111101101110010001110000100101100110011101100001000000110000111010111111100001100010010110010000100101111100101101010100011000111111111000101000111101011111111101100000011010111111010000010001100111110110100001010010010101110001100100100100101000100110111100011011110001001101011011100101111000000010110111010111001111101100101000101110011001111110001000101011010011111110110111100110101100000101101110111110001101101101010011111100001000111101110011010000110001111010011111010000101010101100111110010011000110101111110110100110000110001001000110000011101100010011011001001110001001011110110010111100011011001100110111111000100001010101010100100101100010101101111110010001100110010011111100001110100000110001100110011101010011101110011000110000000111011; sel = 8'b10000101; // Set input values
        #period;
        if (!(out === 4'b1010)) begin
            $display("Mismatch at index 2: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b1110100111101011111101101101001100001111110100101000111100011111000111001101100111100111001110010011011011100101100000010110110110111111000001010000000001111110000010010000110011011011000100100001111011110010111011010011110110111011100000100101101001110111011001000101011111101101110010001110000100101100110011101100001000000110000111010111111100001100010010110010000100101111100101101010100011000111111111000101000111101011111111101100000011010111111010000010001100111110110100001010010010101110001100100100100101000100110111100011011110001001101011011100101111000000010110111010111001111101100101000101110011001111110001000101011010011111110110111100110101100000101101110111110001101101101010011111100001000111101110011010000110001111010011111010000101010101100111110010011000110101111110110100110000110001001000110000011101100010011011001001110001001011110110010111100011011001100110111111000100001010101010100100101100010101101111110010001100110010011111100001110100000110001100110011101010011101110011000110000000111011, 8'b10000101, out, 4'b1010);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 2 passed!");
        end

        // Tick 3: Inputs = 1024'b0101010111110110101011011010101100010101000010010000101100101010111010000111010000001100110100000010001000110001111111110100010000111100111100010001100101111001111011111011111010010100110111111101101010001010111000101011010110110010100111111011011001100101111011010101001101101100110110101111011010000010111000101110110100010100110011111100000100101001001011010010100011011011010110101111001100001001000110101110011011100101100110110011011011001011011110111111100011111101111101110010001000101001000011010100010010101111110110000101011001011111110110011101001010010010101100111001011110011001101010000010111111010001100010111011010010100011001110011001011000010111011100111001001101111101101111000010011001111101001101011001100111111010110011100010111111110010100111001100011100011010000011001000111011000011001111110011100010000110001011000001010101100011010110000001010100001100101011110010101010011111111100101010111000111111001001001000101101001011010010010010110111011010010110010101101110111100000101001000100001111000, 8'b00001110, Generated = out, Reference = 4'b1101
        in = 1024'b0101010111110110101011011010101100010101000010010000101100101010111010000111010000001100110100000010001000110001111111110100010000111100111100010001100101111001111011111011111010010100110111111101101010001010111000101011010110110010100111111011011001100101111011010101001101101100110110101111011010000010111000101110110100010100110011111100000100101001001011010010100011011011010110101111001100001001000110101110011011100101100110110011011011001011011110111111100011111101111101110010001000101001000011010100010010101111110110000101011001011111110110011101001010010010101100111001011110011001101010000010111111010001100010111011010010100011001110011001011000010111011100111001001101111101101111000010011001111101001101011001100111111010110011100010111111110010100111001100011100011010000011001000111011000011001111110011100010000110001011000001010101100011010110000001010100001100101011110010101010011111111100101010111000111111001001001000101101001011010010010010110111011010010110010101101110111100000101001000100001111000; sel = 8'b00001110; // Set input values
        #period;
        if (!(out === 4'b1101)) begin
            $display("Mismatch at index 3: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b0101010111110110101011011010101100010101000010010000101100101010111010000111010000001100110100000010001000110001111111110100010000111100111100010001100101111001111011111011111010010100110111111101101010001010111000101011010110110010100111111011011001100101111011010101001101101100110110101111011010000010111000101110110100010100110011111100000100101001001011010010100011011011010110101111001100001001000110101110011011100101100110110011011011001011011110111111100011111101111101110010001000101001000011010100010010101111110110000101011001011111110110011101001010010010101100111001011110011001101010000010111111010001100010111011010010100011001110011001011000010111011100111001001101111101101111000010011001111101001101011001100111111010110011100010111111110010100111001100011100011010000011001000111011000011001111110011100010000110001011000001010101100011010110000001010100001100101011110010101010011111111100101010111000111111001001001000101101001011010010010010110111011010010110010101101110111100000101001000100001111000, 8'b00001110, out, 4'b1101);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 3 passed!");
        end

        // Tick 4: Inputs = 1024'b1001011010010000000001000010110100010100011111001101100100101000010001000000000110001101100010001101101001101110101110101011010000110100100110000000011101101001011110111101001001100001111101110000001001110100100110110000010010100011000001110001101001000110010010101001001101110001100101010101101100010111001011011011011001100101001110110100100111001010101110110100010111100010011101101011011010100100001001100110110101101100101100001011011111011001101001101111110011011110010011010110110111001011011010011101101110101100101101111100101001011001100000100011111100101100000001000100100111000110010111011001001101001010011101001011111110010100110111000010101111000100101110000011110011010001100001110111100110011100000011101000101000111000010110110110111110111001101101101000010100110001001101000000101010110011110110010111011001100111001001110111100111101001010011100010101100001110111011010101011011100001111100010000001011000011111111101101111101110010111111011100110101011110101111001001101001101110010111011010110111011100, 8'b11000111, Generated = out, Reference = 4'b1010
        in = 1024'b1001011010010000000001000010110100010100011111001101100100101000010001000000000110001101100010001101101001101110101110101011010000110100100110000000011101101001011110111101001001100001111101110000001001110100100110110000010010100011000001110001101001000110010010101001001101110001100101010101101100010111001011011011011001100101001110110100100111001010101110110100010111100010011101101011011010100100001001100110110101101100101100001011011111011001101001101111110011011110010011010110110111001011011010011101101110101100101101111100101001011001100000100011111100101100000001000100100111000110010111011001001101001010011101001011111110010100110111000010101111000100101110000011110011010001100001110111100110011100000011101000101000111000010110110110111110111001101101101000010100110001001101000000101010110011110110010111011001100111001001110111100111101001010011100010101100001110111011010101011011100001111100010000001011000011111111101101111101110010111111011100110101011110101111001001101001101110010111011010110111011100; sel = 8'b11000111; // Set input values
        #period;
        if (!(out === 4'b1010)) begin
            $display("Mismatch at index 4: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b1001011010010000000001000010110100010100011111001101100100101000010001000000000110001101100010001101101001101110101110101011010000110100100110000000011101101001011110111101001001100001111101110000001001110100100110110000010010100011000001110001101001000110010010101001001101110001100101010101101100010111001011011011011001100101001110110100100111001010101110110100010111100010011101101011011010100100001001100110110101101100101100001011011111011001101001101111110011011110010011010110110111001011011010011101101110101100101101111100101001011001100000100011111100101100000001000100100111000110010111011001001101001010011101001011111110010100110111000010101111000100101110000011110011010001100001110111100110011100000011101000101000111000010110110110111110111001101101101000010100110001001101000000101010110011110110010111011001100111001001110111100111101001010011100010101100001110111011010101011011100001111100010000001011000011111111101101111101110010111111011100110101011110101111001001101001101110010111011010110111011100, 8'b11000111, out, 4'b1010);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 4 passed!");
        end

        // Tick 5: Inputs = 1024'b1111110100101000111001001111101000111100111011010010101101111001111001111100001110110110110011111110110110001101100000001101101100111111010110101001101101111110001111000000001111111111011110000100001101100001010101111000011001101110010111110000111111011100001111101001100110000011011111011101101101000110000110101011011000010011001001011001111100100110011000000011100100100001110000000100101100100111001101111001011000011011100001110110000100110111110111001111000000000000101110011000110100100100111101100001101001101010100011100000010111010101011111010100011101111001111110100010111100111010101100110101111001011101011100011001100110111010101110011111010100000100011100111011100001010101110001000111000000110011010011101010011101100110100111101011011111000110001111011110110100110100000010001101101001000011001101010110011110000110100011100011011110010000000111000001010010011110000001110010100111111110101001111010011011111101000011100100000101000101000111001000010001110111111001000000100010010111010111001001110000101110, 8'b01100001, Generated = out, Reference = 4'b1011
        in = 1024'b1111110100101000111001001111101000111100111011010010101101111001111001111100001110110110110011111110110110001101100000001101101100111111010110101001101101111110001111000000001111111111011110000100001101100001010101111000011001101110010111110000111111011100001111101001100110000011011111011101101101000110000110101011011000010011001001011001111100100110011000000011100100100001110000000100101100100111001101111001011000011011100001110110000100110111110111001111000000000000101110011000110100100100111101100001101001101010100011100000010111010101011111010100011101111001111110100010111100111010101100110101111001011101011100011001100110111010101110011111010100000100011100111011100001010101110001000111000000110011010011101010011101100110100111101011011111000110001111011110110100110100000010001101101001000011001101010110011110000110100011100011011110010000000111000001010010011110000001110010100111111110101001111010011011111101000011100100000101000101000111001000010001110111111001000000100010010111010111001001110000101110; sel = 8'b01100001; // Set input values
        #period;
        if (!(out === 4'b1011)) begin
            $display("Mismatch at index 5: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b1111110100101000111001001111101000111100111011010010101101111001111001111100001110110110110011111110110110001101100000001101101100111111010110101001101101111110001111000000001111111111011110000100001101100001010101111000011001101110010111110000111111011100001111101001100110000011011111011101101101000110000110101011011000010011001001011001111100100110011000000011100100100001110000000100101100100111001101111001011000011011100001110110000100110111110111001111000000000000101110011000110100100100111101100001101001101010100011100000010111010101011111010100011101111001111110100010111100111010101100110101111001011101011100011001100110111010101110011111010100000100011100111011100001010101110001000111000000110011010011101010011101100110100111101011011111000110001111011110110100110100000010001101101001000011001101010110011110000110100011100011011110010000000111000001010010011110000001110010100111111110101001111010011011111101000011100100000101000101000111001000010001110111111001000000100010010111010111001001110000101110, 8'b01100001, out, 4'b1011);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 5 passed!");
        end

        // Tick 6: Inputs = 1024'b1110001110110111101011101100011100110101101000001100100101101011000010011111111101000001000100110110010100100011010001011100101001100100011001001110001111001000101111001100111110101000011110011001110001101101111001100011100001001111011101011111111110011110110001101011010111110100100011010001010101011010000111010010101000010101001011111011010100101010110011111101011011000000100111111111001100110100011001101110011000000111000010111011100100001110110100001100101010001100101000010101010010011110111111011010100111010100010010111000000010101000001001110101100011010001010011101100010101010000000101101000101011110110001000101110011011101100101110011000110001000010011100110010010110110010011110110100101111001100000000011011010010011000111000101110010101110100110001010110000010110001011101011100000110010100100110101000101000101001100110101011010010001000001101010111101010001100010110011111010110101000011000111001011001010000010000110111011110010001100001101101000011110101011110001010000100001011100101000000100100010111, 8'b10110110, Generated = out, Reference = 4'b0101
        in = 1024'b1110001110110111101011101100011100110101101000001100100101101011000010011111111101000001000100110110010100100011010001011100101001100100011001001110001111001000101111001100111110101000011110011001110001101101111001100011100001001111011101011111111110011110110001101011010111110100100011010001010101011010000111010010101000010101001011111011010100101010110011111101011011000000100111111111001100110100011001101110011000000111000010111011100100001110110100001100101010001100101000010101010010011110111111011010100111010100010010111000000010101000001001110101100011010001010011101100010101010000000101101000101011110110001000101110011011101100101110011000110001000010011100110010010110110010011110110100101111001100000000011011010010011000111000101110010101110100110001010110000010110001011101011100000110010100100110101000101000101001100110101011010010001000001101010111101010001100010110011111010110101000011000111001011001010000010000110111011110010001100001101101000011110101011110001010000100001011100101000000100100010111; sel = 8'b10110110; // Set input values
        #period;
        if (!(out === 4'b0101)) begin
            $display("Mismatch at index 6: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b1110001110110111101011101100011100110101101000001100100101101011000010011111111101000001000100110110010100100011010001011100101001100100011001001110001111001000101111001100111110101000011110011001110001101101111001100011100001001111011101011111111110011110110001101011010111110100100011010001010101011010000111010010101000010101001011111011010100101010110011111101011011000000100111111111001100110100011001101110011000000111000010111011100100001110110100001100101010001100101000010101010010011110111111011010100111010100010010111000000010101000001001110101100011010001010011101100010101010000000101101000101011110110001000101110011011101100101110011000110001000010011100110010010110110010011110110100101111001100000000011011010010011000111000101110010101110100110001010110000010110001011101011100000110010100100110101000101000101001100110101011010010001000001101010111101010001100010110011111010110101000011000111001011001010000010000110111011110010001100001101101000011110101011110001010000100001011100101000000100100010111, 8'b10110110, out, 4'b0101);
            mismatch_count = mismatch_count + 1;
        end

        else begin
            $display("Test 6 passed!");
        end

        // Tick 7: Inputs = 1024'b0010010110110111010111110100101110010110100001001110000000101101000101000100010011011111001010000100100100111110010001011001001001001101111100111000000110011011110101110011111110110100101011100100010001100101111001111000100000110101110011011011111101101011110001111110100001010110100011111011101010110001010010000111010100011011011000001110010100110110011001001100100000111101110010011111011110000010100100001110111110000101111001010001111000001011111111011000101101101010111110110100011101100101101010011000111011010000100101011010100010100001010101001010100001111001101010010110010101000011110011111100101011110010010010011010010011100100010000100100111111001101100001001101111011001110010111101011110100011001010001010010000100110010111110010010011110010100111100100111110101101101111101011111101011000011001100111001000010000110001111111011101100111011011111111101101000100110100110101011010001001001001011111101001110010010010111001000001010010101101110010110001000010110101010111100010001011101000001011001110110111010, 8'b11000010, Generated = out, Reference = 4'b1111
        in = 1024'b0010010110110111010111110100101110010110100001001110000000101101000101000100010011011111001010000100100100111110010001011001001001001101111100111000000110011011110101110011111110110100101011100100010001100101111001111000100000110101110011011011111101101011110001111110100001010110100011111011101010110001010010000111010100011011011000001110010100110110011001001100100000111101110010011111011110000010100100001110111110000101111001010001111000001011111111011000101101101010111110110100011101100101101010011000111011010000100101011010100010100001010101001010100001111001101010010110010101000011110011111100101011110010010010011010010011100100010000100100111111001101100001001101111011001110010111101011110100011001010001010010000100110010111110010010011110010100111100100111110101101101111101011111101011000011001100111001000010000110001111111011101100111011011111111101101000100110100110101011010001001001001011111101001110010010010111001000001010010101101110010110001000010110101010111100010001011101000001011001110110111010; sel = 8'b11000010; // Set input values
        #period;
        if (!(out === 4'b1111)) begin
            $display("Mismatch at index 7: Inputs = [%b, %b], Generated = [%b], Reference = [%b]", 1024'b0010010110110111010111110100101110010110100001001110000000101101000101000100010011011111001010000100100100111110010001011001001001001101111100111000000110011011110101110011111110110100101011100100010001100101111001111000100000110101110011011011111101101011110001111110100001010110100011111011101010110001010010000111010100011011011000001110010100110110011001001100100000111101110010011111011110000010100100001110111110000101111001010001111000001011111111011000101101101010111110110100011101100101101010011000111011010000100101011010100010100001010101001010100001111001101010010110010101000011110011111100101011110010010010011010010011100100010000100100111111001101100001001101111011001110010111101011110100011001010001010010000100110010111110010010011110010100111100100111110101101101111101011111101011000011001100111001000010000110001111111011101100111011011111111101101000100110100110101011010001001001001011111101001110010010010111001000001010010101101110010110001000010110101010111100010001011101000001011001110110111010, 8'b11000010, out, 4'b1111);
            mismatch_count = mismatch_count + 1;
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