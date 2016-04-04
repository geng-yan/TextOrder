require 'nn'
require 'nngraph'

local CNN = {}
function CNN.cnn(output_size)
  -- there will be 2*n+1 inputs
  local inputs = {}

  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  x = inputs[1]
  local proj1 = nn.Linear(4096,1000)(x)
  local proj2 = nn.ReLU()(proj1)
  local proj3 = nn.Linear(1000,500)(proj2)
  local proj4 = nn.ReLU()(proj3)
  local proj5 = nn.Linear(500,300)(proj4)
  local outputs = {}
  table.insert(outputs,proj5)
  return nn.gModule(inputs, outputs)
end

return CNN

