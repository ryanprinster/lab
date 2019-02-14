
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local timeout = require 'decorators.timeout'
local themes = require 'themes.themes'
local texture_sets = require 'themes.texture_sets'
local pickups_spawn = require 'dmlab.system.pickups_spawn'
local api = {}

local MAP_ENTITIES = [[
*********
*       *
*       *
*       *
*   P   *
*       *
*       *
*       *
*********
]]



local function getRandomRewardStart()
  math.randomseed(os.time())

  local x = math.random(100,800)
  local y = math.random(100,800)

  -- Try again if too close to the agent
  while ((x < 500) and (x > 400)) or ((y < 500) and (y > 400)) do
    x = math.random(100,800)
    y = math.random(100,800)
  end

  reward_origin = tostring(x) .. ' ' .. tostring(y) .. ' 30'

  print(reward_origin)
  return reward_origin
end

local function _respondToEvent()
  pickups_spawn:spawn{
      classname = 'goal',
      origin = getRandomRewardStart(),
      count = '10',
      -- type = pickups.type.GOAL,
  }
end

count = 0
function api:hasEpisodeFinished(_)
  count = count + 1
  if count == 60 then
    count = 0
  end

  return false
end

function api:registerDynamicItems()
  return {'goal'}
end

function api:init(params)
  local my_theme = themes.fromTextureSet{
    textureSet = texture_sets.INVISIBLE_WALLS,
    decalFrequency = 0.0
  }

  make_map.seedRng(1)
  api._map = make_map.makeMap{
      mapName = "empty_room",
      mapEntityLayer = MAP_ENTITIES,
      useSkybox = true,
      theme = my_theme,
      pickups = {
        K = 'goal',
        S = 'strawberry_reward'
      }
  }

end

function api:rewardOverride(kwargs)
    print("REWARD!")
    return nil
end

-- Setting details for keys
function api:createPickup(className)
  return pickups.defaults[className]
end

-- function api:hasEpisodeFinished(_)
--   return episodeFinished
-- end

function api:nextMap()
  return self._map
end

function api:updateSpawnVars(spawnVars)
  _respondToEvent()
  if spawnVars.classname == "info_player_start" then
    -- Spawn facing East.
    spawnVars.angle = "0"
    spawnVars.randomAngleRange = "0"
  end
  return spawnVars

end

timeout.decorate(api, 60 * 60)
custom_observations.decorate(api)

return api