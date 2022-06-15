`timescale 1ns / 1ps

module microprocessor_test;

	// Inputs
	reg CLKin;
	reg reset;

	// Outputs
	wire [6:0] seg_h;
	wire [6:0] seg_l;
	wire [7:0] instruction;
	wire [7:0] read_address;
	wire [7:0] sign_extended_rd;
	wire CLK;
	wire [1:0] write_register;
	wire [7:0] reg_write_data; 
	wire [7:0] read_data_one;
	wire [7:0] read_data_two;
	wire [7:0] ALU_second_input;
	wire [7:0] ALU_output;
	wire [7:0] read_data_memory;
	wire [7:0] PC_in;
//	wire [7:0] PC_out;
	wire [7:0] MUX_0;
	wire [7:0] MUX_1;

	// Instantiate the Unit Under Test (UUT)
	microprocessor uut (
		.CLKin(CLKin), 
		.reset(reset), 
		.seg_h(seg_h), 
		.seg_l(seg_l),
		.instruction(instruction),
		.read_address(read_address),
		.sign_extended_rd(sign_extended_rd),
		.CLK(CLK),
		.write_register(write_register),
		.reg_write_data(reg_write_data),
		.read_data_one(read_data_one),
		.read_data_two(read_data_two),
		.ALU_second_input(ALU_second_input),
		.ALU_output(ALU_output),
		.read_data_memory(read_data_memory),
		.PC_in(PC_in),
//		.PC_out(PC_out),
		.MUX_0(MUX_0),
		.MUX_1(MUX_1)
	);
	
	always #10 CLKin = ~CLKin;
	
	initial begin
		// Initialize Inputs
		CLKin = 1;
		reset = 0;
		#2000
		reset = 1;
		#500
		reset = 0;
	end
      
endmodule

