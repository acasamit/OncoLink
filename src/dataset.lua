local dataset = {}

local function shuffle(t)
	for i = #t, 2, -1 do
		local j = math.random(i)
		t[i], t[j] = t[j], t[i]
	end
end

local function write_lines(lines, split_index, train, val)
	for i, line in ipairs(lines) do
		local id_index = string.find(line, ",")
		local line_cpy = string.sub(line, id_index + 1)

		if i <= split_index then
			assert(train:write(line_cpy .. "\n"), "Failed to write into data_train.csv")
		else
			assert(val:write(line_cpy .. "\n"), "Failed to write into data_val.csv")
		end
	end
end

function dataset.reload()
	os.remove("../data_train.csv")
	os.remove("../data_val.csv")

	local data = assert(io.open("../data.csv", "r"), "Failed to open data.csv")

	local train = assert(io.open("../data_train.csv", "w"), "Failed to create data_train.csv")
	local val = assert(io.open("../data_val.csv", "w"), "Failed to create data_val.csv")

	math.randomseed(os.time())

	local M_lines = {}
	local B_lines = {}

	for line in data:lines() do
		local first_char = string.sub(line, 1, 1)

		if first_char == "M" then
			table.insert(M_lines, line)
		else
			table.insert(B_lines, line)
		end
	end
	data:close()

	shuffle(M_lines)
	shuffle(B_lines)

	local split_M = math.floor(#M_lines * 0.8)
	local split_B = math.floor(#B_lines * 0.8)

	write_lines(M_lines, split_M, train, val)
	write_lines(B_lines, split_B, train, val)

	train:close()
	val:close()
end

return dataset
