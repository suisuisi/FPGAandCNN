//author: shvlad
//email: shvladspb@gmail.com
// synopsys translate_off
`timescale 1 ns / 100 ps
// synopsys translate_on
//`default_nettype none
module cam_wrp  	
(         
	input wire 					rst_n,
	input wire		[7:0]		data_cam,
	input wire					HREF_cam,
	input wire					PCLK_cam,	
	input wire					ctrl_busy,
	output wire		[15:0]	input_fifo_to_sdram,
	output reg 		[23:0]	addr_sdram,
	output reg					wr_enable
);
	reg		[15: 0]		data2fifo;	
	reg					sh_HREF_cam;
	reg 						wr_fifo;
	
	wire		[9:0]		output_rdusedw;
	
	always @( posedge PCLK_cam or negedge rst_n )
		if  ( ! rst_n )
			begin
				sh_HREF_cam	<= 1'h0;
			end			
		else
			begin
				sh_HREF_cam		<= HREF_cam;			
			end	

	always @( posedge PCLK_cam or negedge rst_n )
		if ( !rst_n )
			data2fifo	<= 16'h0;
		else
			if ( wr_fifo )
				data2fifo[7:0]	<= data_cam;
			else
				data2fifo[15:8]	<= data_cam;					

	always @( posedge PCLK_cam or negedge rst_n )
		if ( !rst_n )
			wr_fifo		<= 1'h0;
		else
			if ( HREF_cam )
				wr_fifo	<= !wr_fifo;
			else
				wr_fifo	<= 1'h0;
		
reg [8:0] sh_write;
initial sh_write=9'd0;
reg wr_enable_fifo=1'b0;

always @(posedge ctrl_busy or negedge rst_n)
	begin
		if (!rst_n)
			begin
				sh_write=9'd0;
				wr_enable<=1'b1;
				wr_enable_fifo=1'b1;
				addr_sdram=24'd0;
			end
		else
			begin
				if ((output_rdusedw>600)&&(sh_write==9'd0)) wr_enable<=1'b1;
				if ((output_rdusedw<600)&&(sh_write==9'd320)) wr_enable<=1'b0;
				if (wr_enable)
					begin
						if (sh_write!=0)
							begin
								if (addr_sdram<24'd76799)	addr_sdram=addr_sdram+1'b1;
								else	addr_sdram=24'd0;
							end
						if (sh_write<9'd320) sh_write=sh_write+1'b1;
						else 		sh_write=9'd0;
					end
			end
	end


	
fifo_1024x16 input_fifo 
(	
	.aclr					  ( !rst_n							),	
	.data					  ( data2fifo						),	
	.rdclk              ( ctrl_busy						),	
	.rdreq              ( wr_enable && (sh_write<=9'd319)			),	
	.wrclk              ( PCLK_cam						), 
	.wrreq              ( !wr_fifo && sh_HREF_cam	),	
	.q                  ( input_fifo_to_sdram			),	
	.rdempty            ( 									),
	.rdusedw            ( output_rdusedw				),				
	.wrfull             ( 									),
	.wrusedw            ( 									)
);

endmodule


