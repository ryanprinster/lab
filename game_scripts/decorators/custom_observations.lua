--[[ Copyright (C) 2018 Google Inc.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
]]

local debug_observations = require 'decorators.debug_observations'
local game = require 'dmlab.system.game'
local inventory = require 'common.inventory'
local timeout = require 'decorators.timeout'
local tensor = require 'dmlab.system.tensor'

local obs = {}
local obsSpec = {}
local instructionObservation = ''

local entityLayer = nil

local custom_observations = {}

custom_observations.playerNames = {''}
custom_observations.playerInventory = {}
custom_observations.playerTeams = {}

function custom_observations.addSpec(name, type, shape, callback)
  -- Only add spec if not already present.
  if obs[name] == nil then
    obsSpec[#obsSpec + 1] = {name = name, type = type, shape = shape}
    obs[name] = callback
  end
end

local function velocity()
  local info = game:playerInfo()
  local a = info.angles[2] / 180.0 * math.pi
  local s = math.sin(a)
  local c = math.cos(a)
  local velx = info.vel[1] * c + info.vel[2] * s
  local vely = -info.vel[1] * s + info.vel[2] * c
  return tensor.DoubleTensor{velx, vely, info.vel[3]}
end

local function angularVelocity()
  return tensor.DoubleTensor(game:playerInfo().anglesVel)
end

local function languageChannel()
  return instructionObservation or ''
end

local function teamScore()
  local info = game:playerInfo()
  return tensor.DoubleTensor{info.teamScore, info.otherTeamScore}
end

local framesRemainingTensor = tensor.DoubleTensor(1)
local function framesRemainingAt60()
  framesRemainingTensor:val(timeout.timeRemainingSeconds() * 60)
end

local function position()
  local posinfo = game:playerInfo().pos
  return tensor.DoubleTensor{posinfo[1], posinfo[2], posinfo[3]}
end

local function angles()
  local angleinfo = game:playerInfo().angles
  return tensor.DoubleTensor{angleinfo[1], angleinfo[2], angleinfo[3]}
end

-- Helper function for distanceToClosestWall()
local function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- Helper function for distanceToClosestWall()
local function getCharAtIndex(entityLayer, forward_index, left_index)
  -- Gets the character at an index. Returns '*' if not a valid index.

  lines = {}
  for s in entityLayer:gmatch("[^\r\n]+") do
      table.insert(lines, s)
  end

  if (left_index > tablelength(lines)) or (left_index <= 0) then
    return '*'
  elseif (forward_index > string.len(lines[left_index])) or (forward_index <= 0) then
    return '*'
  else
    return lines[left_index]:sub(forward_index, forward_index)
  end
end

-- Helper function for distanceToClosestWall()
local function getNeighbors(entityLayer, forward_index, left_index)

  neighbors = {}
  neighbors['forward'] = getCharAtIndex(entityLayer, forward_index+1, left_index)
  neighbors['backward'] = getCharAtIndex(entityLayer, forward_index-1, left_index)
  neighbors['right'] = getCharAtIndex(entityLayer, forward_index, left_index+1)
  neighbors['left'] = getCharAtIndex(entityLayer, forward_index, left_index-1)

  neighbors['forward_right'] = getCharAtIndex(entityLayer, forward_index+1, left_index+1)
  neighbors['backward_right'] = getCharAtIndex(entityLayer, forward_index-1, left_index+1)
  neighbors['forward_left'] = getCharAtIndex(entityLayer, forward_index+1, left_index-1)
  neighbors['backward_left'] = getCharAtIndex(entityLayer, forward_index-1, left_index-1)
  return neighbors
end

-- Helper function for distanceToClosestWall()
local function getPlayerIndex()
  posinfo = game:playerInfo().pos
  forward_index = math.floor(posinfo[1] / 100) + 1
  left_index = math.floor(posinfo[2] / 100) + 1
  return forward_index, left_index
end

local function getClosestNeighborAndDistance()
  -- ONLY if the wall is in one of the neighboring grid units 
  -- :P sorry coding gods for hardcoding everything

  -- NOTE: For some stupid reason:
  -- The 'forward' axis is going right on lua entity layer, increasing in index,
  -- and going forward from the default spawn direction.
  -- The 'left' axis is going up on lua entity layer, decreasing in index, 
  -- but going RIGHT FROM THE DEFAULT SPAWN DIRECTION.
  --
  -- Therefore: this code refers to the lua direction. When redering in
  -- deepmind lab, this left and right are reversed.

  forward_index, left_index = getPlayerIndex()
  neighbors = getNeighbors(entityLayer, forward_index, left_index)

  posinfo = game:playerInfo().pos
  forward_loc = posinfo[1]
  left_loc = posinfo[2]
  
  distances = {}

  -- Calculate the distance to the side of the current cell
  distance_backward = forward_loc % 100
  distance_left = left_loc % 100
  distance_forward = 100 - distance_backward
  distance_right = 100 - distance_left

  -- TODO: agent acts like a square in the environment roughly
  distance_diag_fr = (distance_forward^2 + distance_right^2)^.5
  distance_diag_br = (distance_backward^2 + distance_right^2)^.5
  distance_diag_fl = (distance_forward^2 + distance_left^2)^.5
  distance_diag_bl = (distance_backward^2 + distance_left^2)^.5

  if neighbors['forward'] == '*' then
    distances['forward'] = distance_forward
  end
  if neighbors['backward'] == '*' then
    distances['backward'] = distance_backward
  end
  if neighbors['right'] == '*' then
    distances['right'] = distance_right
  end
  if neighbors['left'] == '*' then
    distances['left'] = distance_left
  end
  if neighbors['forward_right'] == '*' then
    distances['forward_right'] = distance_diag_fr
  end
  if neighbors['backward_right'] == '*' then
    distances['backward_right'] = distance_diag_br
  end
  if neighbors['forward_left'] == '*' then
    distances['forward_left'] = distance_diag_fl
  end
  if neighbors['backward_left'] == '*' then
    distances['backward_left'] = distance_diag_bl
  end

  local closest_wall = 'none'
  local closest_wall_distance = math.huge
  for k, v in pairs(distances) do
    if v < closest_wall_distance then
      closest_wall = k
      closest_wall_distance = v
    end
  end
  return closest_wall, closest_wall_distance
end

local function distanceToClosestWall()  
  closest_wall, closest_wall_distance = getClosestNeighborAndDistance()
  -- The agent operates as a square in the environment. It actually
  -- is stopped by the wall at 16.125 units horizontally, and 
  -- sqrt(2)*16.125 units diagonally.
  return tensor.DoubleTensor{closest_wall_distance - 16.125}
end

local function normalizedYaw(angle)
  -- Not expecting input angles > 360 or < -360. 
  -- Sorry again coding gods
  if angle > 180 then
    return angle - 360
  elseif angle <= -180 then
    return angle + 360
  else
    return angle
  end
end

local function angleToClosestWall()
  local absoluteYaw = game:playerInfo().angles[2]

  relativeYaw = absoluteYaw

  if closest_wall == 'forward' then
    relativeYaw = normalizedYaw(relativeYaw)
  elseif closest_wall == 'backward' then
    relativeYaw = normalizedYaw(180 - relativeYaw)
  elseif closest_wall == 'right' then
    relativeYaw = normalizedYaw(90 - relativeYaw)
  elseif closest_wall == 'left' then
    relativeYaw = -normalizedYaw(-90 - relativeYaw)
  else
    relativeYaw = 1000.0 -- must return doubles at all times
  end

  return tensor.DoubleTensor{relativeYaw}

  -- NOTE: Angle is currently only for the non-diagonal
end

--[[ Decorate the api to support custom observations:

1.  Player translational velocity (VEL.TRANS).
2.  Player angular velocity (VEL.ROT).
3.  Language channel for, e.g. giving instructions to the agent (INSTR).
4.  See debug_observations.lua for those.
5.  Player position (POS)
]]
function custom_observations.decorate(api)
  local init = api.init
  print("api.getEntityLayer():")
  if api.getEntityLayer ~= nil then
    entityLayer = api.getEntityLayer()
  else
    print("Must provide getEntityLayer() in lua script")
  end
  print(entityLayer)

  function api:init(params)
    custom_observations.addSpec('VEL.TRANS', 'Doubles', {3}, velocity)
    custom_observations.addSpec('VEL.ROT', 'Doubles', {3}, angularVelocity)
    custom_observations.addSpec('INSTR', 'String', {0}, languageChannel)
    custom_observations.addSpec('TEAM.SCORE', 'Doubles', {2}, teamScore)
    custom_observations.addSpec('FRAMES_REMAINING_AT_60', 'Doubles', {1},
                                framesRemainingAt60)
    custom_observations.addSpec('POS', 'Doubles', {3}, position)
    custom_observations.addSpec('ANGLES', 'Doubles', {3}, angles)    
    custom_observations.addSpec('DISTANCE_TO_WALL', 'Doubles', {1}, 
                                distanceToClosestWall)
    custom_observations.addSpec('ANGLE_TO_WALL', 'Doubles', {1}, 
                                angleToClosestWall)

    api.setInstruction('')
    debug_observations.extend(custom_observations)
    if params.enableCameraMovement == 'true' then
      debug_observations.enableCameraMovement(api)
    end
    return init and init(api, params)
  end

  local customObservationSpec = api.customObservationSpec
  function api:customObservationSpec()
    local specs = customObservationSpec and customObservationSpec(api) or {}
    for i, spec in ipairs(obsSpec) do
      specs[#specs + 1] = spec
    end
    return specs
  end

  local team = api.team
  function api:team(playerId, playerName)
    custom_observations.playerNames[playerId] = playerName
    local result = team and team(self, playerId, playerName) or 'p'
    custom_observations.playerTeams[playerId] = result
    return result
  end

  local spawnInventory = api.spawnInventory
  function api:spawnInventory(loadOut)
    local view = inventory.View(loadOut)
    custom_observations.playerInventory[view:playerId()] = view
    return spawnInventory and spawnInventory(self, loadOut)
  end

  local updateInventory = api.updateInventory
  function api:updateInventory(loadOut)
    local view = inventory.View(loadOut)
    custom_observations.playerInventory[view:playerId()] = view
    return updateInventory and updateInventory(self, loadOut)
  end

  local customObservation = api.customObservation
  function api:customObservation(name)
    return obs[name] and obs[name]() or customObservation(api, name)
  end

  -- Levels can call this to define the language channel observation string
  -- returned to the agent.
  function api.setInstruction(text)
    instructionObservation = text
  end
end

return custom_observations
