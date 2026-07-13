export const CHAT_TITLE_PROMPT = `Generate a concise title for this conversation based on the user's first message.
The title should be 3-7 words in sentence case (capitalize only the first word and proper nouns).
Maximum 50 characters. Focus on the main task or topic.
Output only the title text with no quotes, labels, or formatting.

Examples:
Fix login button on mobile
Add OAuth authentication
Debug failing CI tests
Refactor API client error handling

Bad (too vague): Code changes
Bad (too long): Investigate and fix the issue where the login button does not respond on mobile devices
Bad (wrong case): Fix Login Button On Mobile`
