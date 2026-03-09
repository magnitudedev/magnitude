import { capitalize, slugify } from './helpers'

export { capitalize, slugify }

export function titleCase(str: string): string {
  return str.split(' ').map(word => capitalize(word)).join(' ')
}
