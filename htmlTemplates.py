#2b313e
#475063
css = '''
<style>
.chat-message {
    padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; overflow: auto
    filter: blur(20px);
    }
.chat-message.user {
    background: linear-gradient(to left, #262626, #111111);
}
.chat-message.bot {
    background: linear-gradient(to right, #13373a, #0a2023);
}
.chat-message.user .avatar {
  margin-left: -0.3rem;  /* Adjust margin value for desired shift */
}

.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 40px;
  max-height: 40px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 90%;
  padding: 0 0rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://store-images.s-microsoft.com/image/apps.53694.13632283102504182.38df4173-5fe8-485a-aff2-e817af234808.a9d1e8f2-042d-4597-b7a8-5de3c66f7e40?h=464" style="max-height: 48px; max-width: 48px; border-radius: 20%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/peoples-avatars/man-user-circle-icon.png" style="max-height: 48px; max-width: 48px; border-radius: 10%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
