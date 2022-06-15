`timescale 1ns / 1ps

module adder_eight(
    input [7:0] A,
    input [7:0] B,
    output [7:0] O
    );
    
    assign O = A + B;

endmodule
