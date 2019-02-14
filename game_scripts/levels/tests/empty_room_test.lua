
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local timeout = require 'decorators.timeout'
local themes = require 'themes.themes'
local texture_sets = require 'themes.texture_sets'
local api = {}

local MAP_ENTITIES = [[
*********
*       *
* GAAAG *
* A   A *
* A P A *
* A   A*
* GAAAG *
*       *
*********
]]

-- episodeFinished = false

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
        A = 'apple_reward',
        G = 'goal'
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