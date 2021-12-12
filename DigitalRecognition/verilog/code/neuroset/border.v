module border(
    input clk, go,
    input [9:0] i,
    input [4:0] matrix,
    output reg [1:0] prov
);
reg [10:0] j;
always @(posedge clk)
begin
    if (go == 1)
    begin
        prov = 0;
        for (j = 1'b1; j <= matrix; j = j + 1'b1)
        begin
            if ((i == j*matrix-1'b1) && (prov != 2'b10))
                prov = 2'b10;
            if (((i == 0) || (i == j*matrix)) && (prov != 2'b11))
                prov = 2'b11;
        end
    end
    else
        prov = 0;
end
endmodule