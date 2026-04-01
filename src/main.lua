local MACRO = require("MACRO")
local dataset = require("dataset")

local reload_dataset = false

local args = {...}

function help()
	print("-h, --help		display this message")
	print("-r			reload data.csv and recreate data_train and data_val")
	print("--layer			specify the number of hided layer")
	print("--layer-size		specify the size of hided layers")
end

function parse_args()
	for i, arg in ipairs(args) do
		if arg == "-r" then
			reload_dataset = true

		elseif arg == "--layer" then
			MACRO.HIDED_LAYER = arg[i + 1]

		elseif arg == "--layer-size" then
			MACRO.HIDED_LAYER_SIZE = arg[i + 1]

		else
			help()
		end
	end
end

function check_for_reload()
	if reload_dataset or not io.open("../data_train.csv", "r") or not io.open("../data_val.csv", "r") then
		dataset.reload()
	end
end

function main()
	parse_args()

	check_for_reload()
end

main()
