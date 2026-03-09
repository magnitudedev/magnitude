import { useState } from 'react'
import { apiClient } from '../api/client'

export function useAuth() {
  const [token, setToken] = useState<string | null>(null)

  async function login(email: string, password: string) {
    const data = await apiClient<{ token: string }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
    setToken(data.token)
  }

  return { token, login }
}