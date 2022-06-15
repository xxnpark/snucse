`timescale 1ns / 1ps

module microprocessor(
    input CLKin,
	input reset,
	input [7:0] instruction,
	output [7:0] read_address,
	output [6:0] seg_h,
	output [6:0] seg_l
	);

	wire [1:0] op, rs, rt, rd;
	wire [7:0] sign_extended_rd;
	wire [1:0] write_register;
	wire [7:0] reg_write_data, read_data_one, read_data_two;
	wire [7:0] ALU_second_input, ALU_output;
	wire [7:0] read_data_memory;
	wire [7:0] PC_in;
	wire [7:0] PC_out;
	wire [7:0] MUX_0, MUX_1;
	wire CLK;
	wire RegDst, RegWrite, ALUSrc, Branch, MemRead, MemWrite, MemtoReg, ALUOp;
	
	assign op = instruction[7:6];
	assign rs = instruction[5:4];
	assign rt = instruction[3:2];
	assign rd = instruction[1:0];
	assign sign_extended_rd[1:0] = rd;
	assign sign_extended_rd[7:2] = {6{rd[1]}};

	assign MUX_0 = read_address + 1;
	assign MUX_1 = MUX_0 + sign_extended_rd;

	frequency_divider F1(.clkin(CLKin), .clr(reset), .clkout(CLK));
	control_unit C1(.op(op), .RegDst(RegDst), .RegWrite(RegWrite), .ALUSrc(ALUSrc), .Branch(Branch), .MemRead(MemRead), .MemWrite(MemWrite), .MemtoReg(MemtoReg), .ALUOp(ALUOp));
	MUX_two M1(.I0(rt), .I1(rd), .S0(RegDst), .Z(write_register));
	register_module R1(.read_register_one(rs), .read_register_two(rt), .write_register(write_register), .write_data(reg_write_data), .RegWrite(RegWrite), .CLK(CLK), .reset(reset), .read_data_one(read_data_one), .read_data_two(read_data_two)); 
	MUX_eight M2(.I0(read_data_two), .I1(sign_extended_rd), .S0(ALUSrc), .Z(ALU_second_input));
	ALU A1(.A(read_data_one), .B(ALU_second_input), .ALUOp(ALUOp), .O(ALU_output));
	data_memory D1(.address(ALU_output), .write_data(read_data_two), .MemWrite(MemWrite), .MemRead(MemRead), .CLK(CLK), .reset(reset), .read_data(read_data_memory));
	MUX_eight M3(.I0(ALU_output), .I1(read_data_memory), .S0(MemtoReg), .Z(reg_write_data));
	adder_eight AD1(.A(read_address), .B(1), .O(MUX_0));
	adder_eight AD2(.A(MUX_0), .B(sign_extended_rd), .O(MUX_1));
	MUX_eight M4(.I0(MUX_0), .I1(MUX_1), .S0(Branch), .Z(PC_in));
	PC P1(.D(PC_in), .CLK(CLK), .reset(reset), .Q(read_address));
	
	hex_to_7 H1(.hex(reg_write_data[7:4]), .seg(seg_h));
	hex_to_7 H2(.hex(reg_write_data[3:0]), .seg(seg_l));
	 
endmodule
