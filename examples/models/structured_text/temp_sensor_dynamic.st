FUNCTION_BLOCK NeuralNetworkFB

VAR_INPUT
    input_data : ARRAY[0..4] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    (* Layer 0 variables *)
    weights_0 : ARRAY[0..4, 0..9] OF REAL;
    bias_0 : ARRAY[0..9] OF REAL;
    layer_0_output : ARRAY[0..9] OF REAL;

    (* Layer 1 variables *)
    weights_1 : ARRAY[0..9, 0..9] OF REAL;
    bias_1 : ARRAY[0..9] OF REAL;
    layer_1_output : ARRAY[0..9] OF REAL;

    (* Layer 2 variables *)
    weights_2 : ARRAY[0..9, 0..2] OF REAL;
    bias_2 : ARRAY[0..2] OF REAL;
    layer_2_output : ARRAY[0..2] OF REAL;

    (* Temporary computation variables *)
    i, j : INT;
    sum : REAL;
    max_val : REAL;
    exp_sum : REAL;
    initialized : BOOL := FALSE;
END_VAR

(* Initialize weights (one-time setup) *)
IF NOT initialized THEN
    (* Initialize layer 0 weights *)
    weights_0[0,0] := -0.445159;
    weights_0[0,1] := 0.627839;
    weights_0[0,2] := -0.327399;
    weights_0[0,3] := -0.298371;
    weights_0[0,4] := 0.158793;
    weights_0[0,5] := -0.334292;
    weights_0[0,6] := -0.161885;
    weights_0[0,7] := 0.511321;
    weights_0[0,8] := -0.042912;
    weights_0[0,9] := 0.044624;
    weights_0[1,0] := 0.503638;
    weights_0[1,1] := -0.118902;
    weights_0[1,2] := -0.509557;
    weights_0[1,3] := 0.162514;
    weights_0[1,4] := -0.429035;
    weights_0[1,5] := -0.013840;
    weights_0[1,6] := -0.418917;
    weights_0[1,7] := 0.339058;
    weights_0[1,8] := 0.199603;
    weights_0[1,9] := 0.217632;
    weights_0[2,0] := 0.320823;
    weights_0[2,1] := 0.650330;
    weights_0[2,2] := -0.473451;
    weights_0[2,3] := -0.364074;
    weights_0[2,4] := -0.150603;
    weights_0[2,5] := -0.170975;
    weights_0[2,6] := 0.466145;
    weights_0[2,7] := -0.135315;
    weights_0[2,8] := -0.411599;
    weights_0[2,9] := -0.435844;
    weights_0[3,0] := -0.696904;
    weights_0[3,1] := 0.520537;
    weights_0[3,2] := -0.562433;
    weights_0[3,3] := 0.584049;
    weights_0[3,4] := -0.488544;
    weights_0[3,5] := 0.624043;
    weights_0[3,6] := -0.218557;
    weights_0[3,7] := 0.155840;
    weights_0[3,8] := 0.074428;
    weights_0[3,9] := -0.323533;
    weights_0[4,0] := -0.613059;
    weights_0[4,1] := 0.141350;
    weights_0[4,2] := 0.286769;
    weights_0[4,3] := 0.765734;
    weights_0[4,4] := 0.434346;
    weights_0[4,5] := -0.037822;
    weights_0[4,6] := 0.339381;
    weights_0[4,7] := -0.063226;
    weights_0[4,8] := 0.171952;
    weights_0[4,9] := 0.478369;
    bias_0[0] := -0.077634;
    bias_0[1] := -0.486212;
    bias_0[2] := -0.054921;
    bias_0[3] := -2.552983;
    bias_0[4] := 0.013740;
    bias_0[5] := 2.174905;
    bias_0[6] := 2.147376;
    bias_0[7] := 1.386135;
    bias_0[8] := -0.004573;
    bias_0[9] := 1.996361;

    (* Initialize layer 1 weights *)
    weights_1[0,0] := 0.279341;
    weights_1[0,1] := -0.490818;
    weights_1[0,2] := -0.033860;
    weights_1[0,3] := 0.052609;
    weights_1[0,4] := 0.219421;
    weights_1[0,5] := 0.361208;
    weights_1[0,6] := 0.297950;
    weights_1[0,7] := -0.393297;
    weights_1[0,8] := -0.198817;
    weights_1[0,9] := -0.355393;
    weights_1[1,0] := -0.293792;
    weights_1[1,1] := 0.505055;
    weights_1[1,2] := 0.066838;
    weights_1[1,3] := -0.170058;
    weights_1[1,4] := -0.212144;
    weights_1[1,5] := 0.495598;
    weights_1[1,6] := 0.134224;
    weights_1[1,7] := 0.333538;
    weights_1[1,8] := -0.423919;
    weights_1[1,9] := -0.137260;
    weights_1[2,0] := -0.333317;
    weights_1[2,1] := -0.037792;
    weights_1[2,2] := -0.169093;
    weights_1[2,3] := 0.544607;
    weights_1[2,4] := 0.484035;
    weights_1[2,5] := -0.397176;
    weights_1[2,6] := -0.018798;
    weights_1[2,7] := -0.143084;
    weights_1[2,8] := 0.225956;
    weights_1[2,9] := 0.110448;
    weights_1[3,0] := 0.614243;
    weights_1[3,1] := 1.244041;
    weights_1[3,2] := 0.257517;
    weights_1[3,3] := -0.377380;
    weights_1[3,4] := 0.122296;
    weights_1[3,5] := -0.232057;
    weights_1[3,6] := -0.311260;
    weights_1[3,7] := -0.500125;
    weights_1[3,8] := 0.592350;
    weights_1[3,9] := 0.251889;
    weights_1[4,0] := 0.554439;
    weights_1[4,1] := 0.290110;
    weights_1[4,2] := 0.112758;
    weights_1[4,3] := -0.163860;
    weights_1[4,4] := 0.368117;
    weights_1[4,5] := 0.325973;
    weights_1[4,6] := 0.494165;
    weights_1[4,7] := -0.436352;
    weights_1[4,8] := 0.546032;
    weights_1[4,9] := -0.215206;
    weights_1[5,0] := 0.361143;
    weights_1[5,1] := -1.319624;
    weights_1[5,2] := 0.524926;
    weights_1[5,3] := -0.512397;
    weights_1[5,4] := -0.497354;
    weights_1[5,5] := -0.347770;
    weights_1[5,6] := 0.231320;
    weights_1[5,7] := 0.770105;
    weights_1[5,8] := -0.243553;
    weights_1[5,9] := 0.220384;
    weights_1[6,0] := 0.809685;
    weights_1[6,1] := -0.574775;
    weights_1[6,2] := -0.573412;
    weights_1[6,3] := 0.351427;
    weights_1[6,4] := 0.011063;
    weights_1[6,5] := -0.809211;
    weights_1[6,6] := -0.408086;
    weights_1[6,7] := 0.717961;
    weights_1[6,8] := -0.384621;
    weights_1[6,9] := 0.010253;
    weights_1[7,0] := -0.416220;
    weights_1[7,1] := -0.302910;
    weights_1[7,2] := 0.387971;
    weights_1[7,3] := -0.428287;
    weights_1[7,4] := -0.368947;
    weights_1[7,5] := 0.083653;
    weights_1[7,6] := -0.345027;
    weights_1[7,7] := 0.161866;
    weights_1[7,8] := 0.057498;
    weights_1[7,9] := 0.403654;
    weights_1[8,0] := -0.138637;
    weights_1[8,1] := 0.210438;
    weights_1[8,2] := -0.053329;
    weights_1[8,3] := -0.186443;
    weights_1[8,4] := -0.243178;
    weights_1[8,5] := -0.166998;
    weights_1[8,6] := -0.163783;
    weights_1[8,7] := 0.150571;
    weights_1[8,8] := 0.133055;
    weights_1[8,9] := -0.593544;
    weights_1[9,0] := -0.157295;
    weights_1[9,1] := -1.239563;
    weights_1[9,2] := 0.450461;
    weights_1[9,3] := -0.235578;
    weights_1[9,4] := 0.117049;
    weights_1[9,5] := 0.259138;
    weights_1[9,6] := 0.021362;
    weights_1[9,7] := 0.921632;
    weights_1[9,8] := 0.335857;
    weights_1[9,9] := 0.178988;
    bias_1[0] := -0.367424;
    bias_1[1] := -1.955942;
    bias_1[2] := 0.023235;
    bias_1[3] := 0.000000;
    bias_1[4] := -0.049309;
    bias_1[5] := -0.835884;
    bias_1[6] := 0.000000;
    bias_1[7] := 2.006993;
    bias_1[8] := 0.051287;
    bias_1[9] := 0.161465;

    (* Initialize layer 2 weights *)
    weights_2[0,0] := -0.080444;
    weights_2[0,1] := -0.089052;
    weights_2[0,2] := 0.812036;
    weights_2[1,0] := -1.476020;
    weights_2[1,1] := -0.725518;
    weights_2[1,2] := -0.149382;
    weights_2[2,0] := -0.208250;
    weights_2[2,1] := 0.149819;
    weights_2[2,2] := -0.194204;
    weights_2[3,0] := 0.362083;
    weights_2[3,1] := 0.196811;
    weights_2[3,2] := 0.074062;
    weights_2[4,0] := -0.805061;
    weights_2[4,1] := 0.395131;
    weights_2[4,2] := -0.438517;
    weights_2[5,0] := -0.114728;
    weights_2[5,1] := -0.237476;
    weights_2[5,2] := -0.154911;
    weights_2[6,0] := -0.016607;
    weights_2[6,1] := 0.129336;
    weights_2[6,2] := 0.026178;
    weights_2[7,0] := 1.086698;
    weights_2[7,1] := 0.714801;
    weights_2[7,2] := -0.089847;
    weights_2[8,0] := -0.090735;
    weights_2[8,1] := -0.172974;
    weights_2[8,2] := 0.167604;
    weights_2[9,0] := 0.377956;
    weights_2[9,1] := -0.531758;
    weights_2[9,2] := -0.194333;
    bias_2[0] := 1.009167;
    bias_2[1] := 0.425143;
    bias_2[2] := -1.451678;

    initialized := TRUE;
END_IF;

(* Forward pass computation *)
(* Layer 0: Dense (MatMul + Bias) *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 4 DO
        sum := sum + input_data[i] * weights_0[i,j];
    END_FOR;
    layer_0_output[j] := sum + bias_0[j];
END_FOR;
(* ReLU activation *)
FOR i := 0 TO 9 DO
    IF layer_0_output[i] > 0.0 THEN
        layer_0_output[i] := layer_0_output[i];
    ELSE
        layer_0_output[i] := 0.0;
    END_IF;
END_FOR;

(* Layer 1: Dense (MatMul + Bias) *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_0_output[i] * weights_1[i,j];
    END_FOR;
    layer_1_output[j] := sum + bias_1[j];
END_FOR;
(* ReLU activation *)
FOR i := 0 TO 9 DO
    IF layer_1_output[i] > 0.0 THEN
        layer_1_output[i] := layer_1_output[i];
    ELSE
        layer_1_output[i] := 0.0;
    END_IF;
END_FOR;

(* Layer 2: Dense (MatMul + Bias) *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_1_output[i] * weights_2[i,j];
    END_FOR;
    output_data[j] := sum + bias_2[j];
END_FOR;
(* Softmax activation *)
max_val := output_data[0];
FOR i := 1 TO 2 DO
    IF output_data[i] > max_val THEN
        max_val := output_data[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(output_data[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;

END_FUNCTION_BLOCK



PROGRAM prog0
    VAR
        nn : NeuralNetworkFB;
        sensor_input : ARRAY[0..4] OF REAL;
        nn_output : ARRAY[0..2] OF REAL;
        output_regs : ARRAY[0..2] OF REAL;
    END_VAR

    (* Example sensor input values *)
    sensor_input[0] := 23.5;  
    sensor_input[1] := 17.0; 
    sensor_input[2] := 45.0;
    sensor_input[3] := 100.0;   
    sensor_input[4] := 80.0;

    (* Call the neural network function block *)
    nn(input_data := sensor_input);
    nn_output := nn.output_data;

    (* nn_output now contains the classification probabilities *)
    output_regs[0] := nn_output[0];
    output_regs[1] := nn_output[1];
    output_regs[2] := nn_output[2];
    
END_PROGRAM


CONFIGURATION Config0
    RESOURCE Res0 ON PLC
        TASK Main(INTERVAL := T#1000ms,PRIORITY := 0);
        PROGRAM Inst0 WITH Main : prog0;
    END_RESOURCE
END_CONFIGURATION
