css = '''
<style>
/* Define CSS variables for consistent theming */
:root {
    --dark-user: #1f2937;
    --dark-bot: #334155;
    --border-user: #374151;
    --border-bot: #475063;
    --text-color: #fff;
    --pad: 1.5rem;
    --border-radius: 0.75rem;
    --message-radius: 0.5rem;
    --transition-duration: 0.3s;
}

.chat-message {
    padding: var(--pad);
    border-radius: var(--border-radius);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    width: 100%;
    transition: transform var(--transition-duration), box-shadow var(--transition-duration);
}

.chat-message:hover {
    transform: scale(1.02); /* Subtle zoom effect on hover */
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.chat-message.user {
    background-color: var(--dark-user);
    border: 1px solid var(--border-user);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.chat-message.bot {
    background-color: var(--dark-bot);
    border: 1px solid var(--border-bot);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.chat-message .message {
    width: 100%;
    padding: 1rem 1.5rem;
    color: var(--text-color);
    font-size: 1rem;
    border-radius: var(--message-radius);
    line-height: 1.5;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.chat-message.user .message {
    background: linear-gradient(135deg, var(--border-user), var(--dark-user));
}

.chat-message.bot .message {
    background: linear-gradient(135deg, var(--border-bot), var(--dark-bot));
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">🤖 Bot: {{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">👤 User: {{MSG}}</div>
</div>
'''