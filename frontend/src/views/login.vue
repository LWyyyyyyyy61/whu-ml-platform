<template>
    <div class="login-page">
      <h1>Login</h1>
      <form @submit.prevent="login">
        <label>
          Username:
          <input type="text" v-model="username" required>
        </label>
        <label>
          Password:
          <input type="password" v-model="password" required>
        </label>
        <button type="submit">Login</button>
      </form>
    </div>
  </template>
  
  <script>
  export default {
    name: 'LoginPage',
    data() {
      return {
        username: '',
        password: ''
      }
    },
    methods: {
      login() {
        // 在这里添加登录验证逻辑
        if (this.username === 'admin' && this.password === 'password') {
          // 登录成功,存储用户信息并重定向到主页面
          localStorage.setItem('user', JSON.stringify({ name: 'Admin' }));
          this.$router.push('/main');
        } else {
          // 登录失败,显示错误信息
          alert('Invalid username or password');
        }
      }
    }
  }
  </script>
  
  <style scoped>
  .login-page {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
  }
  
  form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  </style>