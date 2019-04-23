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
local custom_observations = require 'decorators.custom_observations'
local game = require 'dmlab.system.game'
local timeout = require 'decorators.timeout'
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

function api:init(params)
  make_map.seedRng(1)
  api._map = make_map.makeMap{
      mapName = "empty_room",
      mapEntityLayer = MAP_ENTITIES,
      useSkybox = true,
  }
end

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

function api:hasEpisodeFinished(_)
  print('Position:')
  print(game:playerInfo().pos[1])
  for k,v in pairs(game:playerInfo().pos) do
    print('k: ', k)
    print('v: ', v)
    print('')
  end
end

-- local customObservationSpec = api.customObservationSpec
-- function api:customObservationSpec()
--   local specs = customObservationSpec and customObservationSpec(api) or {}
--   for i, spec in ipairs(obsSpec) do
--     specs[#specs + 1] = spec
--   end
--   return specs
-- end

-- local observationTable = {
--     LOCATION = tensor.Tensor{0, 0, 0},
-- }

-- local customObservation = api.customObservation
-- function api:customObservation(name)
--   return observationTable[name]
-- end

-- -- See customObservation how to implement these.

-- function api:customObservationSpec()
--   return {
--     {name = 'LOCATION', type = 'doubles', shape = {3}},
--   }
-- end


timeout.decorate(api, 60 * 60)
custom_observations.decorate(api)

return api