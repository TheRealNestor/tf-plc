FUNCTION_BLOCK NeuralNetworkFB

VAR_INPUT
    input_data : ARRAY[0..2] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    (* Layer 0 variables *)
    layer_0_output : ARRAY[0..4] OF REAL;

    (* Layer 1 variables *)
    weights_1 : ARRAY[0..4, 0..9] OF REAL;
    layer_1_output : ARRAY[0..9] OF REAL;

    (* Layer 2 variables *)
    bias_2 : ARRAY[0..9] OF REAL;
    layer_2_output : ARRAY[0..9] OF REAL;

    (* Layer 3 variables *)
    layer_3_output : ARRAY[0..9] OF REAL;

    (* Layer 4 variables *)
    weights_4 : ARRAY[0..9, 0..9] OF REAL;
    layer_4_output : ARRAY[0..9] OF REAL;

    (* Layer 5 variables *)
    bias_5 : ARRAY[0..9] OF REAL;
    layer_5_output : ARRAY[0..9] OF REAL;

    (* Layer 6 variables *)
    layer_6_output : ARRAY[0..9] OF REAL;

    (* Layer 7 variables *)
    weights_7 : ARRAY[0..9, 0..2] OF REAL;
    layer_7_output : ARRAY[0..2] OF REAL;

    (* Layer 8 variables *)
    bias_8 : ARRAY[0..2] OF REAL;
    layer_8_output : ARRAY[0..2] OF REAL;

    (* Layer 9 variables *)
    layer_9_output : ARRAY[0..2] OF REAL;

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

    weights_1[0,0] := -0.445068;
    weights_1[0,1] := 0.627930;
    weights_1[0,2] := -0.327393;
    weights_1[0,3] := -0.298340;
    weights_1[0,4] := 0.158813;
    weights_1[0,5] := -0.334229;
    weights_1[0,6] := -0.161865;
    weights_1[0,7] := 0.511230;
    weights_1[0,8] := -0.042908;
    weights_1[0,9] := 0.044617;
    weights_1[1,0] := 0.503418;
    weights_1[1,1] := -0.118896;
    weights_1[1,2] := -0.509766;
    weights_1[1,3] := 0.162476;
    weights_1[1,4] := -0.428955;
    weights_1[1,5] := -0.013840;
    weights_1[1,6] := -0.418945;
    weights_1[1,7] := 0.339111;
    weights_1[1,8] := 0.199585;
    weights_1[1,9] := 0.217651;
    weights_1[2,0] := 0.320801;
    weights_1[2,1] := 0.650391;
    weights_1[2,2] := -0.473389;
    weights_1[2,3] := -0.364014;
    weights_1[2,4] := -0.150635;
    weights_1[2,5] := -0.171021;
    weights_1[2,6] := 0.466064;
    weights_1[2,7] := -0.135376;
    weights_1[2,8] := -0.411621;
    weights_1[2,9] := -0.435791;
    weights_1[3,0] := -0.696777;
    weights_1[3,1] := 0.520508;
    weights_1[3,2] := -0.562500;
    weights_1[3,3] := 0.583984;
    weights_1[3,4] := -0.488525;
    weights_1[3,5] := 0.624023;
    weights_1[3,6] := -0.218506;
    weights_1[3,7] := 0.155884;
    weights_1[3,8] := 0.074402;
    weights_1[3,9] := -0.323486;
    weights_1[4,0] := -0.613281;
    weights_1[4,1] := 0.141357;
    weights_1[4,2] := 0.286865;
    weights_1[4,3] := 0.765625;
    weights_1[4,4] := 0.434326;
    weights_1[4,5] := -0.037811;
    weights_1[4,6] := 0.339355;
    weights_1[4,7] := -0.063232;
    weights_1[4,8] := 0.171997;
    weights_1[4,9] := 0.478271;

    bias_2[0] := -0.077637;
    bias_2[1] := -0.486328;
    bias_2[2] := -0.054932;
    bias_2[3] := -2.552734;
    bias_2[4] := 0.013741;
    bias_2[5] := 2.175781;
    bias_2[6] := 2.146484;
    bias_2[7] := 1.385742;
    bias_2[8] := -0.004574;
    bias_2[9] := 1.996094;


    weights_4[0,0] := 0.279297;
    weights_4[0,1] := -0.490723;
    weights_4[0,2] := -0.033875;
    weights_4[0,3] := 0.052612;
    weights_4[0,4] := 0.219360;
    weights_4[0,5] := 0.361328;
    weights_4[0,6] := 0.297852;
    weights_4[0,7] := -0.393311;
    weights_4[0,8] := -0.198853;
    weights_4[0,9] := -0.355469;
    weights_4[1,0] := -0.293701;
    weights_4[1,1] := 0.504883;
    weights_4[1,2] := 0.066833;
    weights_4[1,3] := -0.170044;
    weights_4[1,4] := -0.212158;
    weights_4[1,5] := 0.495605;
    weights_4[1,6] := 0.134277;
    weights_4[1,7] := 0.333496;
    weights_4[1,8] := -0.423828;
    weights_4[1,9] := -0.137207;
    weights_4[2,0] := -0.333252;
    weights_4[2,1] := -0.037781;
    weights_4[2,2] := -0.169067;
    weights_4[2,3] := 0.544434;
    weights_4[2,4] := 0.484131;
    weights_4[2,5] := -0.397217;
    weights_4[2,6] := -0.018799;
    weights_4[2,7] := -0.143066;
    weights_4[2,8] := 0.225952;
    weights_4[2,9] := 0.110474;
    weights_4[3,0] := 0.614258;
    weights_4[3,1] := 1.244141;
    weights_4[3,2] := 0.257568;
    weights_4[3,3] := -0.377441;
    weights_4[3,4] := 0.122314;
    weights_4[3,5] := -0.232056;
    weights_4[3,6] := -0.311279;
    weights_4[3,7] := -0.500000;
    weights_4[3,8] := 0.592285;
    weights_4[3,9] := 0.251953;
    weights_4[4,0] := 0.554199;
    weights_4[4,1] := 0.290039;
    weights_4[4,2] := 0.112732;
    weights_4[4,3] := -0.163818;
    weights_4[4,4] := 0.368164;
    weights_4[4,5] := 0.325928;
    weights_4[4,6] := 0.494141;
    weights_4[4,7] := -0.436279;
    weights_4[4,8] := 0.545898;
    weights_4[4,9] := -0.215210;
    weights_4[5,0] := 0.361084;
    weights_4[5,1] := -1.319336;
    weights_4[5,2] := 0.524902;
    weights_4[5,3] := -0.512207;
    weights_4[5,4] := -0.497314;
    weights_4[5,5] := -0.347656;
    weights_4[5,6] := 0.231323;
    weights_4[5,7] := 0.770020;
    weights_4[5,8] := -0.243530;
    weights_4[5,9] := 0.220337;
    weights_4[6,0] := 0.809570;
    weights_4[6,1] := -0.574707;
    weights_4[6,2] := -0.573242;
    weights_4[6,3] := 0.351318;
    weights_4[6,4] := 0.011063;
    weights_4[6,5] := -0.809082;
    weights_4[6,6] := -0.408203;
    weights_4[6,7] := 0.717773;
    weights_4[6,8] := -0.384521;
    weights_4[6,9] := 0.010254;
    weights_4[7,0] := -0.416260;
    weights_4[7,1] := -0.302979;
    weights_4[7,2] := 0.387939;
    weights_4[7,3] := -0.428223;
    weights_4[7,4] := -0.368896;
    weights_4[7,5] := 0.083679;
    weights_4[7,6] := -0.344971;
    weights_4[7,7] := 0.161865;
    weights_4[7,8] := 0.057495;
    weights_4[7,9] := 0.403564;
    weights_4[8,0] := -0.138672;
    weights_4[8,1] := 0.210449;
    weights_4[8,2] := -0.053314;
    weights_4[8,3] := -0.186401;
    weights_4[8,4] := -0.243164;
    weights_4[8,5] := -0.166992;
    weights_4[8,6] := -0.163818;
    weights_4[8,7] := 0.150513;
    weights_4[8,8] := 0.133057;
    weights_4[8,9] := -0.593750;
    weights_4[9,0] := -0.157349;
    weights_4[9,1] := -1.239258;
    weights_4[9,2] := 0.450439;
    weights_4[9,3] := -0.235596;
    weights_4[9,4] := 0.117065;
    weights_4[9,5] := 0.259033;
    weights_4[9,6] := 0.021362;
    weights_4[9,7] := 0.921875;
    weights_4[9,8] := 0.335938;
    weights_4[9,9] := 0.178955;

    bias_5[0] := -0.367432;
    bias_5[1] := -1.956055;
    bias_5[2] := 0.023239;
    bias_5[3] := 0.000000;
    bias_5[4] := -0.049316;
    bias_5[5] := -0.835938;
    bias_5[6] := 0.000000;
    bias_5[7] := 2.007812;
    bias_5[8] := 0.051300;
    bias_5[9] := 0.161499;


    weights_7[0,0] := -0.080444;
    weights_7[0,1] := -0.089050;
    weights_7[0,2] := 0.812012;
    weights_7[1,0] := -1.475586;
    weights_7[1,1] := -0.725586;
    weights_7[1,2] := -0.149414;
    weights_7[2,0] := -0.208252;
    weights_7[2,1] := 0.149780;
    weights_7[2,2] := -0.194214;
    weights_7[3,0] := 0.362061;
    weights_7[3,1] := 0.196777;
    weights_7[3,2] := 0.074036;
    weights_7[4,0] := -0.805176;
    weights_7[4,1] := 0.395020;
    weights_7[4,2] := -0.438477;
    weights_7[5,0] := -0.114746;
    weights_7[5,1] := -0.237427;
    weights_7[5,2] := -0.154907;
    weights_7[6,0] := -0.016602;
    weights_7[6,1] := 0.129395;
    weights_7[6,2] := 0.026184;
    weights_7[7,0] := 1.086914;
    weights_7[7,1] := 0.714844;
    weights_7[7,2] := -0.089844;
    weights_7[8,0] := -0.090759;
    weights_7[8,1] := -0.172974;
    weights_7[8,2] := 0.167603;
    weights_7[9,0] := 0.377930;
    weights_7[9,1] := -0.531738;
    weights_7[9,2] := -0.194336;

    bias_8[0] := 1.008789;
    bias_8[1] := 0.425049;
    bias_8[2] := -1.452148;


    initialized := TRUE;
END_IF;

(* Forward pass computation *)
(* Layer 0: Reshape (copy input to output) *)
FOR i := 0 TO 4 DO
    layer_0_output[i] := input_data[i];
END_FOR;

(* Layer 1: MatMul *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 4 DO
        sum := sum + layer_0_output[i] * weights_1[i,j];
    END_FOR;
    layer_1_output[j] := sum;
END_FOR;

(* Layer 2: Add (Bias) *)
FOR i := 0 TO 9 DO
    layer_2_output[i] := layer_1_output[i] + bias_2[i];
END_FOR;

(* Layer 3: Activation (RELU) *)
FOR i := 0 TO 9 DO
    layer_3_output[i] := MAX(layer_2_output[i], 0.0);
END_FOR;

(* Layer 4: MatMul *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_3_output[i] * weights_4[i,j];
    END_FOR;
    layer_4_output[j] := sum;
END_FOR;

(* Layer 5: Add (Bias) *)
FOR i := 0 TO 9 DO
    layer_5_output[i] := layer_4_output[i] + bias_5[i];
END_FOR;

(* Layer 6: Activation (RELU) *)
FOR i := 0 TO 9 DO
    layer_6_output[i] := MAX(layer_5_output[i], 0.0);
END_FOR;

(* Layer 7: MatMul *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + layer_6_output[i] * weights_7[i,j];
    END_FOR;
    layer_7_output[j] := sum;
END_FOR;

(* Layer 8: Add (Bias) *)
FOR i := 0 TO 2 DO
    layer_8_output[i] := layer_7_output[i] + bias_8[i];
END_FOR;

(* Layer 9: Activation (SOFTMAX) *)
max_val := layer_8_output[0];
FOR i := 1 TO 2 DO
    IF layer_8_output[i] > max_val THEN
        max_val := layer_8_output[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(layer_8_output[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;

END_FUNCTION_BLOCK