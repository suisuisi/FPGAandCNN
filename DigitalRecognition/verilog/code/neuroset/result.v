module result(clk,enable,STOP,memstartp,read_addressp,qp,re,RESULT);

parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_address_pix=0;

input clk,enable;
output reg STOP;
input [SIZE_address_pix-1:0] memstartp;
input [SIZE_1-1:0] qp;
output reg re;
output reg [SIZE_address_pix-1:0] read_addressp;
output reg [3:0] RESULT;

reg [3:0] marker;
reg signed [SIZE_1-1:0] buff;

wire signed [SIZE_1-1:0]  p1;
always @(posedge clk)
begin
if (enable==1)
begin
re=1;
case (marker)
	0: begin read_addressp=memstartp+0; end
	1: begin read_addressp=memstartp+1; buff = 0; end
	2: begin read_addressp=memstartp+2; if (p1>=buff) begin buff=p1; RESULT=0; end end
	3: begin read_addressp=memstartp+3; if (p1>=buff) begin buff=p1; RESULT=1; end end
	4: begin read_addressp=memstartp+4; if (p1>=buff) begin buff=p1; RESULT=2; end end
	5: begin read_addressp=memstartp+5; if (p1>=buff) begin buff=p1; RESULT=3; end end
	6: begin read_addressp=memstartp+6; if (p1>=buff) begin buff=p1; RESULT=4; end end
	7: begin read_addressp=memstartp+7; if (p1>=buff) begin buff=p1; RESULT=5; end end
	8: begin read_addressp=memstartp+8; if (p1>=buff) begin buff=p1; RESULT=6; end end
	9: begin read_addressp=memstartp+9; if (p1>=buff) begin buff=p1; RESULT=7; end end
	10: begin read_addressp=memstartp+10; if (p1>=buff) begin buff=p1; RESULT=8; end end
	11: begin read_addressp=memstartp+11; if (p1>=buff) begin buff=p1; RESULT=9; end end
	12: begin read_addressp=memstartp+12; if (p1>=buff) begin buff=p1; RESULT=10; end STOP=1; end
	default: $display("Check case result");
endcase
marker=marker+1;
end
else 
begin
re=0;
marker=0;
STOP=0;
end
end

assign p1=qp[SIZE_1-1:0];
endmodule
