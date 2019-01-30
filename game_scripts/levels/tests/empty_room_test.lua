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

local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local positionTrigger = require 'common.position_trigger'
local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local game_entities = require 'dmlab.system.game_entities'
local pickups_spawn = require 'dmlab.system.pickups_spawn'
local timeout = require 'decorators.timeout'
local themes = require 'themes.themes'
local texture_sets = require 'themes.texture_sets'
local api = {}


local entityLayer = [[
...***...
....P....
...***...
]]

local variationsLayer = [[
AAA...BBB
AAA...BBB
AAA...BBB
]]

local positionTriggerLayer = [[
LLL...LLL
LLL...LLL
LLL...LLL
]]

local ENTITY_LAYER = [[
*********
****.****
****.****
****.****
*...P...*
****.****
****.****
****.****
*********
]]

local VARIATIONS_LAYER = [[
.........         
....A....         
....A....         
....A....         
.BBB.CCC.         
....D....         
....D....         
....D....         
.........         
]]



local entityLayerH = [[
...*****...
...*****.K.
...*****...
*.*******.*
*.*******.*
*H*******.*
*....P....*
*H*******.*
*.*******.*
*.*******.*
...*****...
...*****.K.
...*****...
]]

local positionTriggerLayerH = [[
LLL........
LLL........
LLL........
...........
...........
...........
...........
...........
...........
...........
LLL........
LLL........
LLL........
]]

-- TODO(prinster): Code Cleanup. Consolidate global vars, helper functions,
-- local variable names

-- Defines which arm the agent is forced to pick, and thereby
-- which key should be picked up. (Sorry, there are no enums in lua)
-- "upperArm" means the upper door is open
-- "lowerArm" means the lower door is open
mazeState = "upperArm"

-- Players position at the time of successfully picking up a reward
player_pos_at_reward = nil

-- Players angle (yaw) at the time of successfully picking up a reward
player_angle_at_reward = nil

-- Whether or not the player tried the incorrect key first, 
-- and thereby whether or not the player needs to go back to the start 
-- box to initiate a new trial
tried_wrong_key = false

mazeStateStreak = 1

-- previous_positions = []
function chooseNextMazeState()
	randnum = math.random(1,2)
	
	-- Don't let same maze repeat more than thrice 
	if mazeStateStreak >= 3 then
		randnum = (randnum % 2) + 1
		mazeStateStreak = 1
	end

	-- Choose random arm
	if randnum == 1 then
		if mazeState == "lowerArm" then
			mazeStateStreak = mazeStateStreak + 1
		end
		mazeState = "lowerArm"
	else
		if mazeState == "upperArm" then
			mazeStateStreak = mazeStateStreak + 1
		end
		mazeState = "upperArm"
	end
end

local myPositionTrigger = positionTrigger.new()
function api:hasEpisodeFinished(_)
	--[[print("Angle: ")
	print(game:playerInfo().angles[1])
	print(game:playerInfo().angles[2])
	print(game:playerInfo().angles[3])--]]
  myPositionTrigger:update(game:playerInfo().pos)
end

local function _respondToEvent()
  print('RESPOND_TO_EVENT()')
  tried_wrong_key = false
  --[[pickups_spawn:spawn{
      classname = 'strawberry_reward',
      origin = '750 450 30',
      count = '5',
  }--]]
end

function api:init(params)
  
  api._doorsOpened = {}
	
  myPositionTrigger:start{
    name = 'test trigger',
    maze = positionTriggerLayerH,
    triggerWhenExit = 'L',
    callback = function() _respondToEvent() return true end,
  }

  local my_theme = themes.fromTextureSet{
    textureSet = texture_sets.INVISIBLE_WALLS,
    decalFrequency = 0.0
  }
  make_map.seedRng(1)
  api._map = make_map.makeMap{
      mapName = "empty_room",
      mapEntityLayer = entityLayerH,
--      mapVariationsLayer = variationsLayer,
      pickups = {
        A = 'apple_reward',
        G = 'goal',
	K = 'key',
      },
      useSkybox = true,
      theme = my_theme,
  }
end


function api:createPickup(className)
  if className == 'key' then 
    return {
      name = 'Key',
      classname = 'key',
      model = 'models/hr_key_lrg.md3',
      count = 50,
      quantity = 1,
      type = pickups.type.GOAL,
    }
  end
  return pickups.defaults[className]
end

function api:nextMap()
	-- Reset after restart
	key_id_count = 1
  	chooseNextMazeState()
	return self._map
end

function api:canPickup(spawnId, _playerId)
	
			print("TRIED WRONG KEY START")
			print(tried_wrong_key)

	-- If the player attempted to pickup the wrong key first, 
	-- player now cannot pick up anything
	if tried_wrong_key == true then
		player_pos_at_reward = game:playerInfo().pos
		player_angle_at_reward = game:playerInfo().angles[2]
		return false
	end



	if mazeState == "lowerArm" then
		-- Player attemps to pick up the incorrect key first
		if tostring(spawnId) == "1" then
			player_pos_at_reward = game:playerInfo().pos
			player_angle_at_reward = game:playerInfo().angles[2]
			
			print("TRIED WRONG KEY START")
			print(tried_wrong_key)
			--assert(false)
			
			return true
		end
	elseif mazeState == "upperArm" then
		-- Player attemps to pick up the incorrect key first
		if tostring(spawnId) == "2" then
			player_pos_at_reward = game:playerInfo().pos
			player_angle_at_reward = game:playerInfo().angles[2]
			print("TRIED WRONG KEY START")
			print(tried_wrong_key)
			--assert(false)
			return true
		end
	end

	print("TRIED WRONG KEY END")
	print(tried_wrong_key)

	-- Reward recieved, save player position
	player_pos_at_reward = game:playerInfo().pos
	player_angle_at_reward = game:playerInfo().angles[2]
	return true
end

function api:rewardOverride(kwargs)
	print("REWARD OVERRIDE")
	--for k, v in kwargs.l
	--print(kwargs.location)
	assert(false)
	if tried_wrong_key == true then
		return 0
	end
	return 1
end

-- Assigning ids to key pickups for use in canPickup
key_id_count = 1
local function assignKeyIds(spawnVars_key)
	print("CALLING ASSIGN KEY IDS")
	assert(spawnVars_key.classname == "key", "spawnVars is not a key")
  	-- Note: For some reason, spawnVars_keyid seems to be a number 
	-- despite the tostring function, no idea why
	
	if spawnVars_key.id == nil then
		spawnVars_key.id = tostring(key_id_count)
		key_id_count = key_id_count + 1
	end	
	return spawnVars_key
end



-- Sets an individual doors initial state
local function setDoorState(spawnVars_door, state)
	assert(state == "open" or state == "closed")
	if state == "open" then
		spawnVars_door.spawnflags = "1"
	elseif state == "closed" then
		-- Sets to default, closed
		spawnVars_door.spawnflags = "0"
	end
	-- Forcing door to hold spawned position
	spawnVars_door.speed = "1"
	spawnVars_door.wait = ".00001"
	return spawnVars_door
end


-- Sets all the door states given a 
local function setMazeDoorState(spawnVars_door)
	assert(spawnVars_door.classname == "func_door")
	
	if mazeState == "upperArm" then	
	  if spawnVars_door.targetname == "door_1_5" then
	  	spawnVars_door = setDoorState(spawnVars_door,"closed")
	  elseif spawnVars_door.targetname == "door_1_7" then
	  	spawnVars_door = setDoorState(spawnVars_door,"open")
	  end
	elseif mazeState == "lowerArm" then
	  if spawnVars_door.targetname == "door_1_5" then
	  	spawnVars_door = setDoorState(spawnVars_door,"open")
	  elseif spawnVars_door.targetname == "door_1_7" then
	  	spawnVars_door = setDoorState(spawnVars_door,"closed")
	  end
  	else 
	  error("invalid mazeState")
	end
	return spawnVars_door
end

local function setPlayerInfo(spawnVars)
	  -- Default origin and angle, conditional on mazeState
	  if mazeState == "lowerArm" then
    		spawnVars.origin = "150 150 30"
    		spawnVars.angle = "90"
    		spawnVars.randomAngleRange = "0"
	  elseif mazeState == "upperArm" then
    		spawnVars.origin = "150 1150 30"
    		spawnVars.angle = "-90"
    		spawnVars.randomAngleRange = "0"
	  else
		-- Default default spawn. Shouldn't be hit...
    		spawnVars.origin = "150 150 30"
    		spawnVars.angle = "90"
    		spawnVars.randomAngleRange = "0"
	  end
	  
	  -- Set spawn position for new episode
	  if player_pos_at_reward ~= nil then
		print(player_pos_at_reward)
		spawnVars.origin = tostring(player_pos_at_reward[1]
				.. ' ' .. player_pos_at_reward[2] 
				.. ' ' .. player_pos_at_reward[3])
	  end
	  
	  -- Set spawn angle for new episode
	  -- NOTE: Currently only setting yaw angle. 
	  -- Unknown if others can be set.
	  if player_angle_at_reward ~= nil then
	  	print(player_angle_at_reward)
	  	spawnVars.angle = player_angle_at_reward .. ''
	  end

	  return spawnVars
end

-- Update any variables at spawn we want to be different from default values
function api:updateSpawnVars(spawnVars)
  
  tried_wrong_key = true
	
  if spawnVars.classname == "key" then
  	spawnVars = assignKeyIds(spawnVars)
  end

  if spawnVars.classname == "func_door" then
	spawnVars = setMazeDoorState(spawnVars)
  end
  
  if spawnVars.classname == "info_player_start" then
	spawnVars = setPlayerInfo(spawnVars)
  end

  return spawnVars
end

function api:gameEvent(eventName,_)
	print("GAME EVENT: " .. eventName)
end


timeout.decorate(api, 60 * 60)
custom_observations.decorate(api)

return api
