`timescale 1ns / 1ps

module IMEM(
    input [7:0] read_address,
	output [7:0] instruction
	);
	
	wire [7:0] MemByte[31:0];
	
	assign MemByte[0]  = {2'b01, 2'b00, 2'b10, 2'b01}; // $s2=Mem[$s0+1] / 1  / $s2 <= 1
	assign MemByte[1]  = {2'b11, 2'b00, 2'b00, 2'b01}; // j+1            / 0  / to 3
	assign MemByte[2]  = {2'b00, 2'b01, 2'b10, 2'b00}; // $s0=$s1+$s2    / X
	assign MemByte[3]  = {2'b10, 2'b10, 2'b10, 2'b01}; // Mem[$s2+1]=$s2 / 2  / Mem[2] <= 1
	assign MemByte[4]  = {2'b01, 2'b00, 2'b11, 2'b01}; // $s3=Mem[$s0+1] / 1  / $s3 <= 1
	assign MemByte[5]  = {2'b01, 2'b10, 2'b00, 2'b01}; // $s0=Mem[$s2+1] / 1  / $s0 <= 1
	assign MemByte[6]  = {2'b10, 2'b00, 2'b01, 2'b01}; // Mem[$s0+1]=$s1 / 2  / Mem[2] <= 0
	assign MemByte[7]  = {2'b00, 2'b00, 2'b00, 2'b00}; // $s0=$s0+$s0    / 2  / $s0 <= 2
	assign MemByte[8]  = {2'b01, 2'b00, 2'b11, 2'b01}; // $s3=Mem[$s0+1] / 3  / $s3 <= 3
	assign MemByte[9]  = {2'b00, 2'b00, 2'b11, 2'b11}; // $s3=$s0+$s3    / 5  / $s3 <= 5
	assign MemByte[10] = {2'b00, 2'b11, 2'b11, 2'b00}; // $s0=$s3+$s3    / 10 = 0A / $s0 <= 10
	assign MemByte[11] = {2'b00, 2'b00, 2'b00, 2'b11}; // $s3=$s0+$s0    / 20 = 14 / $s3 <= 20
	assign MemByte[12] = {2'b01, 2'b11, 2'b10, 2'b11}; // $s2=Mem[$s3-1] / -3 = FD / $s2 <= 11111101
	assign MemByte[13] = {2'b11, 2'b00, 2'b00, 2'b11}; // j-1 / $s0+$s0
	
	assign instruction = MemByte[read_address];


endmodule