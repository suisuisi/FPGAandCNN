module maxp(clk,maxp_en,memstartp,memstartzap,read_addressp,write_addressp,re,we,qp,dp,STOP,matrix2,matrix);

parameter SIZE_1=0;
parameter SIZE_2=0;
parameter SIZE_3=0;
parameter SIZE_4=0;
parameter SIZE_address_pix=0;

input clk,maxp_en;
output reg STOP;
input [SIZE_address_pix-1:0] memstartp,memstartzap;
output reg [SIZE_address_pix-1:0] read_addressp,write_addressp;
output reg re,we;
input signed [SIZE_1-1:0] qp;
output reg signed [SIZE_1-1:0] dp;
input [4:0] matrix;
input [9:0] matrix2;

reg [9:0] i;
reg [9:0] j;
reg [SIZE_1-1:0] buff;
reg [2:0] marker;
wire [9:0] i_wr,i_read;
initial i=0;
initial j=0;
initial marker=0;
always @(posedge clk)
begin
if (maxp_en==1)
    begin
    case (marker)
        0: begin read_addressp=memstartp+i_read; re=1;
                 if ((i!=0)||(j!=0))
                 begin
                 if (qp[SIZE_1-1:0]>buff[SIZE_1-1:0]) buff[SIZE_1-1:0]=qp[SIZE_1-1:0];
                 end
           end
        1: begin read_addressp=memstartp+i_read+1;
                 if ((i!=0)||(j!=0))
                 begin
                 if (qp[SIZE_1-1:0]>buff[SIZE_1-1:0]) dp[SIZE_1-1:0]=qp[SIZE_1-1:0]; else dp[SIZE_1-1:0]=buff[SIZE_1-1:0];
                 write_addressp=memstartzap+i_wr-1;
                 we=1;
                 end
           end
        2: begin read_addressp=memstartp+i_read+matrix; buff=qp; we=0; end
        3: begin read_addressp=memstartp+i_read+matrix+1;
                 if (qp[SIZE_1-1:0]>buff[SIZE_1-1:0]) buff[SIZE_1-1:0]=qp[SIZE_1-1:0];
                 if (i!=matrix-2) begin i=i+2; if (i_read==matrix2) STOP=1; end else begin i=0; j=j+1;  end
           end
        default: $display("Check case MaxPooling");
        endcase
        if (marker!=3) marker=marker+1; else marker=0;    end
else
    begin
        STOP=0;
        re=0;
        we=0;
        i=0;
        j=0;
        marker=0;
    end
end
assign i_wr=(i>>1)+j*(matrix>>1);
assign i_read=i+(matrix+matrix)*j;
endmodule
