import { defaultModel } from '@magnitudedev/tools'
import {
  shellModel, fileReadModel, fileWriteModel, fileEditModel,
  fileTreeModel, fileSearchModel, webSearchModel, webFetchModel,
  agentCreateModel, agentKillModel, skillModel, browserActionModel,
} from '.'
import type { StateModel } from '@magnitudedev/tools'

// Use type-erased StateModel for the registry since we store heterogeneous models
type AnyStateModel = StateModel<any, any, any, any, any>

const models = new Map<string, AnyStateModel>()
models.set('shell', shellModel as AnyStateModel)
models.set('fileRead', fileReadModel as AnyStateModel)
models.set('fileWrite', fileWriteModel as AnyStateModel)
models.set('fileEdit', fileEditModel as AnyStateModel)
models.set('fileTree', fileTreeModel as AnyStateModel)
models.set('fileSearch', fileSearchModel as AnyStateModel)
models.set('webSearch', webSearchModel as AnyStateModel)
models.set('webFetch', webFetchModel as AnyStateModel)
models.set('agentCreate', agentCreateModel as AnyStateModel)
models.set('agentKill', agentKillModel as AnyStateModel)
models.set('skill', skillModel as AnyStateModel)
models.set('click', browserActionModel as AnyStateModel)
models.set('doubleClick', browserActionModel as AnyStateModel)
models.set('rightClick', browserActionModel as AnyStateModel)
models.set('type', browserActionModel as AnyStateModel)
models.set('scroll', browserActionModel as AnyStateModel)
models.set('drag', browserActionModel as AnyStateModel)
models.set('navigate', browserActionModel as AnyStateModel)
models.set('goBack', browserActionModel as AnyStateModel)
models.set('switchTab', browserActionModel as AnyStateModel)
models.set('newTab', browserActionModel as AnyStateModel)
models.set('screenshot', browserActionModel as AnyStateModel)
models.set('evaluate', browserActionModel as AnyStateModel)

export function getModelForToolKey(toolKey: string): AnyStateModel {
  return models.get(toolKey) ?? defaultModel
}
