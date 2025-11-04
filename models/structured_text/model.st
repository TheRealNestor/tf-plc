FUNCTION_BLOCK NeuralNetworkFB

VAR_INPUT
    input_data : ARRAY[0..2] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    (* Layer 0 variables *)
    weights_0 : ARRAY[0..4, 0..9] OF REAL;
    layer_0_output : ARRAY[0..9] OF REAL;

    (* Layer 1 variables *)
    bias_1 : ARRAY[0..9] OF REAL;
    layer_1_output : ARRAY[0..9] OF REAL;

    (* Layer 2 variables *)
    layer_2_output : ARRAY[0..9] OF REAL;

    (* Layer 3 variables *)
    weights_3 : ARRAY[0..9, 0..9] OF REAL;
    layer_3_output : ARRAY[0..9] OF REAL;

    (* Layer 4 variables *)
    bias_4 : ARRAY[0..9] OF REAL;
    layer_4_output : ARRAY[0..9] OF REAL;

    (* Layer 5 variables *)
    layer_5_output : ARRAY[0..9] OF REAL;

    (* Layer 6 variables *)
    weights_6 : ARRAY[0..9, 0..2] OF REAL;
    layer_6_output : ARRAY[0..2] OF REAL;

    (* Layer 7 variables *)
    bias_7 : ARRAY[0..2] OF REAL;
    layer_7_output : ARRAY[0..2] OF REAL;

    (* Layer 8 variables *)
    layer_8_output : ARRAY[0..2] OF REAL;

    (* Temporary computation variables *)
    i : INT;
    j : INT;
    sum : REAL;
    max_val : REAL;
    exp_sum : REAL;
    initialized : BOOL := FALSE;
END_VAR

(* Initialize weights (one-time setup) *)
IF NOT initialized THEN
    (* Initialize layer 0 weights *)
    weights_0[0,0] := -0.445068;
    weights_0[0,1] := 0.627930;
    weights_0[0,2] := -0.327393;
    weights_0[0,3] := -0.298340;
    weights_0[0,4] := 0.158813;
    weights_0[0,5] := -0.334229;
    weights_0[0,6] := -0.161865;
    weights_0[0,7] := 0.511230;
    weights_0[0,8] := -0.042908;
    weights_0[0,9] := 0.044617;
    weights_0[1,0] := 0.503418;
    weights_0[1,1] := -0.118896;
    weights_0[1,2] := -0.509766;
    weights_0[1,3] := 0.162476;
    weights_0[1,4] := -0.428955;
    weights_0[1,5] := -0.013840;
    weights_0[1,6] := -0.418945;
    weights_0[1,7] := 0.339111;
    weights_0[1,8] := 0.199585;
    weights_0[1,9] := 0.217651;
    weights_0[2,0] := 0.320801;
    weights_0[2,1] := 0.650391;
    weights_0[2,2] := -0.473389;
    weights_0[2,3] := -0.364014;
    weights_0[2,4] := -0.150635;
    weights_0[2,5] := -0.171021;
    weights_0[2,6] := 0.466064;
    weights_0[2,7] := -0.135376;
    weights_0[2,8] := -0.411621;
    weights_0[2,9] := -0.435791;
    weights_0[3,0] := -0.696777;
    weights_0[3,1] := 0.520508;
    weights_0[3,2] := -0.562500;
    weights_0[3,3] := 0.583984;
    weights_0[3,4] := -0.488525;
    weights_0[3,5] := 0.624023;
    weights_0[3,6] := -0.218506;
    weights_0[3,7] := 0.155884;
    weights_0[3,8] := 0.074402;
    weights_0[3,9] := -0.323486;
    weights_0[4,0] := -0.613281;
    weights_0[4,1] := 0.141357;
    weights_0[4,2] := 0.286865;
    weights_0[4,3] := 0.765625;
    weights_0[4,4] := 0.434326;
    weights_0[4,5] := -0.037811;
    weights_0[4,6] := 0.339355;
    weights_0[4,7] := -0.063232;
    weights_0[4,8] := 0.171997;
    weights_0[4,9] := 0.478271;

    (* Initialize layer 1 weights *)
    bias_1[0] := -0.077637;
    bias_1[1] := -0.486328;
    bias_1[2] := -0.054932;
    bias_1[3] := -2.552734;
    bias_1[4] := 0.013741;
    bias_1[5] := 2.175781;
    bias_1[6] := 2.146484;
    bias_1[7] := 1.385742;
    bias_1[8] := -0.004574;
    bias_1[9] := 1.996094;

    (* Initialize layer 2 weights *)

    (* Initialize layer 3 weights *)
    weights_3[0,0] := 0.279297;
    weights_3[0,1] := -0.490723;
    weights_3[0,2] := -0.033875;
    weights_3[0,3] := 0.052612;
    weights_3[0,4] := 0.219360;
    weights_3[0,5] := 0.361328;
    weights_3[0,6] := 0.297852;
    weights_3[0,7] := -0.393311;
    weights_3[0,8] := -0.198853;
    weights_3[0,9] := -0.355469;
    weights_3[1,0] := -0.293701;
    weights_3[1,1] := 0.504883;
    weights_3[1,2] := 0.066833;
    weights_3[1,3] := -0.170044;
    weights_3[1,4] := -0.212158;
    weights_3[1,5] := 0.495605;
    weights_3[1,6] := 0.134277;
    weights_3[1,7] := 0.333496;
    weights_3[1,8] := -0.423828;
    weights_3[1,9] := -0.137207;
    weights_3[2,0] := -0.333252;
    weights_3[2,1] := -0.037781;
    weights_3[2,2] := -0.169067;
    weights_3[2,3] := 0.544434;
    weights_3[2,4] := 0.484131;
    weights_3[2,5] := -0.397217;
    weights_3[2,6] := -0.018799;
    weights_3[2,7] := -0.143066;
    weights_3[2,8] := 0.225952;
    weights_3[2,9] := 0.110474;
    weights_3[3,0] := 0.614258;
    weights_3[3,1] := 1.244141;
    weights_3[3,2] := 0.257568;
    weights_3[3,3] := -0.377441;
    weights_3[3,4] := 0.122314;
    weights_3[3,5] := -0.232056;
    weights_3[3,6] := -0.311279;
    weights_3[3,7] := -0.500000;
    weights_3[3,8] := 0.592285;
    weights_3[3,9] := 0.251953;
    weights_3[4,0] := 0.554199;
    weights_3[4,1] := 0.290039;
    weights_3[4,2] := 0.112732;
    weights_3[4,3] := -0.163818;
    weights_3[4,4] := 0.368164;
    weights_3[4,5] := 0.325928;
    weights_3[4,6] := 0.494141;
    weights_3[4,7] := -0.436279;
    weights_3[4,8] := 0.545898;
    weights_3[4,9] := -0.215210;
    weights_3[5,0] := 0.361084;
    weights_3[5,1] := -1.319336;
    weights_3[5,2] := 0.524902;
    weights_3[5,3] := -0.512207;
    weights_3[5,4] := -0.497314;
    weights_3[5,5] := -0.347656;
    weights_3[5,6] := 0.231323;
    weights_3[5,7] := 0.770020;
    weights_3[5,8] := -0.243530;
    weights_3[5,9] := 0.220337;
    weights_3[6,0] := 0.809570;
    weights_3[6,1] := -0.574707;
    weights_3[6,2] := -0.573242;
    weights_3[6,3] := 0.351318;
    weights_3[6,4] := 0.011063;
    weights_3[6,5] := -0.809082;
    weights_3[6,6] := -0.408203;
    weights_3[6,7] := 0.717773;
    weights_3[6,8] := -0.384521;
    weights_3[6,9] := 0.010254;
    weights_3[7,0] := -0.416260;
    weights_3[7,1] := -0.302979;
    weights_3[7,2] := 0.387939;
    weights_3[7,3] := -0.428223;
    weights_3[7,4] := -0.368896;
    weights_3[7,5] := 0.083679;
    weights_3[7,6] := -0.344971;
    weights_3[7,7] := 0.161865;
    weights_3[7,8] := 0.057495;
    weights_3[7,9] := 0.403564;
    weights_3[8,0] := -0.138672;
    weights_3[8,1] := 0.210449;
    weights_3[8,2] := -0.053314;
    weights_3[8,3] := -0.186401;
    weights_3[8,4] := -0.243164;
    weights_3[8,5] := -0.166992;
    weights_3[8,6] := -0.163818;
    weights_3[8,7] := 0.150513;
    weights_3[8,8] := 0.133057;
    weights_3[8,9] := -0.593750;
    weights_3[9,0] := -0.157349;
    weights_3[9,1] := -1.239258;
    weights_3[9,2] := 0.450439;
    weights_3[9,3] := -0.235596;
    weights_3[9,4] := 0.117065;
    weights_3[9,5] := 0.259033;
    weights_3[9,6] := 0.021362;
    weights_3[9,7] := 0.921875;
    weights_3[9,8] := 0.335938;
    weights_3[9,9] := 0.178955;

    (* Initialize layer 4 weights *)
    bias_4[0] := -0.367432;
    bias_4[1] := -1.956055;
    bias_4[2] := 0.023239;
    bias_4[3] := 0.000000;
    bias_4[4] := -0.049316;
    bias_4[5] := -0.835938;
    bias_4[6] := 0.000000;
    bias_4[7] := 2.007812;
    bias_4[8] := 0.051300;
    bias_4[9] := 0.161499;

    (* Initialize layer 5 weights *)

    (* Initialize layer 6 weights *)
    weights_6[0,0] := -0.080444;
    weights_6[0,1] := -0.089050;
    weights_6[0,2] := 0.812012;
    weights_6[1,0] := -1.475586;
    weights_6[1,1] := -0.725586;
    weights_6[1,2] := -0.149414;
    weights_6[2,0] := -0.208252;
    weights_6[2,1] := 0.149780;
    weights_6[2,2] := -0.194214;
    weights_6[3,0] := 0.362061;
    weights_6[3,1] := 0.196777;
    weights_6[3,2] := 0.074036;
    weights_6[4,0] := -0.805176;
    weights_6[4,1] := 0.395020;
    weights_6[4,2] := -0.438477;
    weights_6[5,0] := -0.114746;
    weights_6[5,1] := -0.237427;
    weights_6[5,2] := -0.154907;
    weights_6[6,0] := -0.016602;
    weights_6[6,1] := 0.129395;
    weights_6[6,2] := 0.026184;
    weights_6[7,0] := 1.086914;
    weights_6[7,1] := 0.714844;
    weights_6[7,2] := -0.089844;
    weights_6[8,0] := -0.090759;
    weights_6[8,1] := -0.172974;
    weights_6[8,2] := 0.167603;
    weights_6[9,0] := 0.377930;
    weights_6[9,1] := -0.531738;
    weights_6[9,2] := -0.194336;

    (* Initialize layer 7 weights *)
    bias_7[0] := 1.008789;
    bias_7[1] := 0.425049;
    bias_7[2] := -1.452148;

    (* Initialize layer 8 weights *)

    initialized := TRUE;
END_IF;

(* Forward pass computation *)
(* Layer 0: MatMul *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 4 DO
        sum := sum + input_data[i] * weights_0[i,j];
    END_FOR;
    layer_0_output[j] := sum;
END_FOR;

(* Layer 1: Add (Bias) *)
FOR i := 0 TO 9 DO
    layer_1_output[i] := layer_0_output[i] + bias_1[i];
END_FOR;

(* Layer 2: Activation (RELU) *)
FOR i := 0 TO 9 DO
    layer_2_output[i] := MAX(layer_1_output[i], 0.0);
END_FOR;

(* Layer 3: MatMul *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_2_output[i] * weights_3[i,j];
    END_FOR;
    layer_3_output[j] := sum;
END_FOR;

(* Layer 4: Add (Bias) *)
FOR i := 0 TO 9 DO
    layer_4_output[i] := layer_3_output[i] + bias_4[i];
END_FOR;

(* Layer 5: Activation (RELU) *)
FOR i := 0 TO 9 DO
    layer_5_output[i] := MAX(layer_4_output[i], 0.0);
END_FOR;

(* Layer 6: MatMul *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_5_output[i] * weights_6[i,j];
    END_FOR;
    layer_6_output[j] := sum;
END_FOR;

(* Layer 7: Add (Bias) *)
FOR i := 0 TO 2 DO
    layer_7_output[i] := layer_6_output[i] + bias_7[i];
END_FOR;

(* Layer 8: Activation (SOFTMAX) *)
max_val := layer_7_output[0];
FOR i := 1 TO 2 DO
    IF layer_7_output[i] > max_val THEN
        max_val := layer_7_output[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(layer_7_output[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;

END_FUNCTION_BLOCK