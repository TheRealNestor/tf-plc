FUNCTION_BLOCK NeuralNetworkFB

VAR_INPUT
    input_data : ARRAY[0..4] OF SINT;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..9] OF REAL;
END_VAR

VAR CONSTANT
    weights_1 : ARRAY[0..49] OF REAL := [-0.444482, 0.629847, -0.327717, -0.295441, 0.157719, -0.334133, -0.161499, 0.511321, -0.042132, 0.045200, 0.504844, -0.117776, -0.509290, 0.162794, -0.430842, -0.014741, -0.418430, 0.338197, 0.200938, 0.218468, 0.318271, 0.650330, -0.473861, -0.361764, -0.150025, -0.171980, 0.466145, -0.136889, -0.411599, -0.436935, -0.696904, 0.522313, -0.562433, 0.584852, -0.488544, 0.624043, -0.220226, 0.157020, 0.074542, -0.323935, -0.614592, 0.143380, 0.287859, 0.765734, 0.434689, -0.039310, 0.337680, -0.064418, 0.171770, 0.478369];

    weights_4 : ARRAY[0..99] OF REAL := [0.280521, -0.488365, -0.031605, 0.051459, 0.219306, 0.363189, 0.299612, -0.391875, -0.200560, -0.355192, -0.293272, 0.509146, 0.067726, -0.171530, -0.211473, 0.496996, 0.132296, 0.333819, -0.424440, -0.135534, -0.331525, -0.041563, -0.167057, 0.544607, 0.485606, -0.395048, -0.019455, -0.145139, 0.223880, 0.112166, 0.612045, 1.246889, 0.257358, -0.377365, 0.121401, -0.229383, -0.311285, -0.500729, 0.592350, 0.252373, 0.554666, 0.290941, 0.112876, -0.162953, 0.368120, 0.324959, 0.494165, -0.435416, 0.545708, -0.214985, 0.363402, -1.319624, 0.523746, -0.510301, -0.497354, -0.350446, 0.229573, 0.769236, -0.242537, 0.219658, 0.809685, -0.571491, -0.573412, 0.351636, 0.011749, -0.809211, -0.408562, 0.718437, -0.382462, 0.009347, -0.414406, -0.301332, 0.388295, -0.428824, -0.368120, 0.082833, -0.346305, 0.159653, 0.055970, 0.401928, -0.140260, 0.207815, -0.054181, -0.184394, -0.242803, -0.165665, -0.163425, 0.152396, 0.135261, -0.593544, -0.159387, -1.236498, 0.451505, -0.235853, 0.117485, 0.261241, 0.019455, 0.921632, 0.335821, 0.177596];

    weights_7 : ARRAY[0..29] OF REAL := [-0.081355, -0.091404, 0.812036, -1.476020, -0.725518, -0.147062, -0.209200, 0.148531, -0.191819, 0.360288, 0.194233, 0.076728, -0.801932, 0.394179, -0.441185, -0.116222, -0.239935, -0.153456, -0.011622, 0.131393, 0.025576, 1.092487, 0.714092, -0.089516, -0.092978, -0.171382, 0.166244, 0.383533, -0.531285, -0.191819];

    bias_2 : ARRAY[0..9] OF REAL := [-0.077728, -0.486146, -0.055262, -2.552989, 0.013622, 2.175047, 2.147089, 1.386354, -0.004372, 1.996328];

    bias_5 : ARRAY[0..9] OF REAL := [-0.366862, -1.956535, 0.023151, 0.000000, -0.049086, -0.836756, 0.000000, 2.007301, 0.051818, 0.161092];

    bias_8 : ARRAY[0..2] OF REAL := [1.010356, 0.425502, -1.451074];

END_VAR

VAR
    (* Layer 0 output variable *)
    layer_0_output : ARRAY[0..4] OF SINT;

    (* Layer 1 output variable *)
    layer_1_output : ARRAY[0..9] OF REAL;

    (* Layer 4 output variable *)
    layer_4_output : ARRAY[0..9] OF REAL;

    (* Layer 7 output variable *)
    layer_7_output : ARRAY[0..2] OF REAL;

    (* Layer 9 output variable *)
    layer_9_output : ARRAY[0..2] OF REAL;

    (* Layer 2 output variable *)
    layer_2_output : ARRAY[0..9] OF REAL;

    (* Layer 5 output variable *)
    layer_5_output : ARRAY[0..9] OF REAL;

    (* Layer 8 output variable *)
    layer_8_output : ARRAY[0..2] OF REAL;

    (* Layer 3 output variable *)
    layer_3_output : ARRAY[0..9] OF REAL;

    (* Layer 6 output variable *)
    layer_6_output : ARRAY[0..9] OF REAL;

    (* Temporary computation variables *)
    i : DINT;
    j : DINT;
    sum : REAL;
    max_val : REAL;
    exp_sum : REAL;

END_VAR

(* Forward pass computation *)
(* Layer 0: Reshape (copy input to output) *)
FOR i := 0 TO 4 DO
    layer_0_output[i] := input_data[i];
END_FOR;

(* Layer 1: MatMul *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 4 DO
        sum := sum + input_data[i] * weights_1[i * 10 + j];
    END_FOR;
    layer_1_output[j] := sum;
END_FOR;

(* Layer 4: MatMul *)
FOR j := 0 TO 9 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + input_data[i] * weights_4[i * 10 + j];
    END_FOR;
    layer_4_output[j] := sum;
END_FOR;

(* Layer 7: MatMul *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 9 DO
        sum := sum + input_data[i] * weights_7[i * 3 + j];
    END_FOR;
    layer_7_output[j] := sum;
END_FOR;

(* Layer 9: Activation (SOFTMAX) *)
max_val := input_data[0];
FOR i := 1 TO 2 DO
    IF input_data[i] > max_val THEN
        max_val := input_data[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    layer_9_output[i] := EXP(input_data[i] - max_val);
    exp_sum := exp_sum + layer_9_output[i];
END_FOR;

FOR i := 0 TO 2 DO
    layer_9_output[i] := layer_9_output[i] / exp_sum;
END_FOR;

(* Layer 2: Add (Bias) *)
FOR i := 0 TO 9 DO
    layer_2_output[i] := layer_1_output[i] + bias_2[i];
END_FOR;

(* Layer 5: Add (Bias) *)
FOR i := 0 TO 9 DO
    layer_5_output[i] := layer_4_output[i] + bias_5[i];
END_FOR;

(* Layer 8: Add (Bias) *)
FOR i := 0 TO 2 DO
    layer_8_output[i] := layer_7_output[i] + bias_8[i];
END_FOR;

(* Layer 3: Activation (RELU) *)
FOR i := 0 TO 9 DO
    layer_3_output[i] := MAX(layer_2_output[i], 0.0);
END_FOR;

(* Layer 6: Activation (RELU) *)
FOR i := 0 TO 9 DO
    layer_6_output[i] := MAX(layer_5_output[i], 0.0);
END_FOR;

END_FUNCTION_BLOCK
