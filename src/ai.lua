local ai = {}

local MACRO = require("MACRO")

local layers = {}

local function split(str)
	local result = {}
	for value in string.gmatch(str, "([^,]+)") do
		table.insert(result, tonumber(value))
	end
	return result
end

local function softmax(output_layer)
	local max_z = -math.huge

	for _, n in ipairs(output_layer) do
		if n.z > max_z then max_z = n.z end
	end

	local sum = 0
	local exps = {}

	for i, n in ipairs(output_layer) do
		exps[i] = math.exp(n.z - max_z)
		sum = sum + exps[i]
	end

	for i, n in ipairs(output_layer) do
		n.activation = exps[i] / sum
	end
end

local function xavier_init(n_in, n_out)
	local limit = math.sqrt(6 / (n_in + n_out))
	return (math.random() * 2 - 1) * limit
end

function setup_neural_network()
	for i = 1, MACRO.HIDED_LAYER, 1 do
		layers[i] = {}

		for j = 1, MACRO.HIDED_LAYER_SIZE, 1 do
			table.insert(layers[i], {
				z = 0,
				weight = {},
				biase = 0
			})

			local size = i == 1 and MACRO.INPUT_LAYER or MACRO.HIDED_LAYER_SIZE
			for k = 1, size, 1 do
				table.insert(layers[i][j].weight, xavier_init(size, MACRO.HIDED_LAYER_SIZE))
			end
		end

	end

	layers[#layers + 1] = {}
	local output_neurons

	for i = 1, MACRO.OUTPUT_LAYER, 1 do
		table.insert(layers[#layers], {
			z = 0,
			weight = {},
			biase = 0
		})

		for j = 1, MACRO.HIDED_LAYER_SIZE, 1 do
			table.insert(layers[#layers][i].weight, xavier_init(MACRO.HIDED_LAYER_SIZE, MACRO.OUTPUT_LAYER))
		end
	end
end

local function feedforward(input, column)
	for i = 1, #layers[column], 1 do
		layers[column][i].z = 0

		for j, val in ipairs(input) do
			layers[column][i].z = layers[column][i].z + val * layers[column][i].weight[j]
		end

		layers[column][i].z = layers[column][i].z + layers[column][i].biase

		if column ~= #layers then
			layers[column][i].activation = math.max(0, layers[column][i].z)
		end
	end
end

local function backdrop(Loss, P, result, input)
	for i = #layers, 1, -1 do -- layer

		for j = 1, #layers[i], 1 do -- neuron
			if i ~= #layers then
				local sum = 0

				for n = 1, #layers[i + 1] do
					sum = sum + (layers[i + 1][n].delta * layers[i + 1][n].weight[j])
				end

				layers[i][j].delta = sum * (layers[i][j].z > 0 and 1 or 0)
			else
				local target = (j == 1) and result or (1 - result)
				layers[i][j].delta = P[j] - target
			end

			for k = 1, #layers[i][j].weight, 1 do -- weight
				local gradient

				if i == 1 then
					gradient = layers[i][j].delta * input[k]
				else
					gradient = layers[i][j].delta * layers[i - 1][k].activation
				end

				layers[i][j].weight[k] = layers[i][j].weight[k] - (MACRO.LEARNING_RATE * gradient)
			end

			layers[i][j].biase = layers[i][j].biase - (MACRO.LEARNING_RATE * layers[i][j].delta)
		end
	end
end

local function epoch()
	local train = assert(io.open("../data_train.csv", "r"), "Failed to open data_train.csv")
	local result

	local loss_sum = 0
	local epoch_n = 0

	for line in train:lines() do
		local r_index = string.find(line, ",")
		local first_input = split(string.sub(line, r_index + 1))
		local input = first_input

		result = string.sub(line, 1, 1)
		result = result == "B" and 0 or 1

		for column, _ in ipairs(layers) do
			feedforward(input, column)

			if column == #layers then
				softmax(layers[column])
				local P = {}
				for n, _ in ipairs(layers[column]) do table.insert(P, layers[column][n].activation) end

				local Loss = -((result * math.log(math.max(P[1], 1e-15))) + (1 - result) * math.log(math.max(1- P[1], 1e-15))) -- BCE
				loss_sum = loss_sum + Loss
				epoch_n = epoch_n + 1
				backdrop(Loss, P, result, first_input)
			end

			input = {}
			for i, _ in ipairs(layers[column]) do
				table.insert(input, layers[column][i].activation)
			end
		end
	end

	print(loss_sum/epoch_n)
	train:close()
end

function shuffle_file(input_path, output_path)
	local f = assert(io.open(input_path, "r"), "Failed to open input file")

	local lines = {}
	for line in f:lines() do
		table.insert(lines, line)
	end
	f:close()

	for i = #lines, 2, -1 do
		local j = math.random(i)
		lines[i], lines[j] = lines[j], lines[i]
	end

	local out = assert(io.open(output_path, "w"), "Failed to open output file")
	for _, line in ipairs(lines) do
		out:write(line .. "\n")
	end
	out:close()
end

function ai.start_train()
	setup_neural_network()

	for i = 1, MACRO.EPOCH, 1 do
		print("Epoch "..i.."/"..MACRO.EPOCH..":")

		epoch()
		shuffle_file("../data_train.csv", "../data_train.csv")
	end
end

return ai
