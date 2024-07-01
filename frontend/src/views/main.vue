<template>
    <div class="main-page">
      <header>
        <nav>
          <ul>
            <li><router-link to="/dashboard">Dashboard</router-link></li>
            <li><router-link to="/tasks">Tasks</router-link></li>
            <li><router-link to="/settings">Settings</router-link></li>
          </ul>
        </nav>
        <div class="user-info">
          <span>Welcome, {{ user.name }}</span>
          <button @click="logout">Logout</button>
        </div>
      </header>
  
      <main>
        <router-view></router-view>
      </main>
    </div>
  </template>
  
  <script>
  export default {
    name: 'MainPage',
    data() {
      return {
        user: JSON.parse(localStorage.getItem('user'))
      }
    },
    methods: {
      logout() {
        // 清除用户信息并重定向到登录页面
        localStorage.removeItem('user');
        this.$router.push('/');
      }
    }
  }
  </script>
  
  <style scoped>
  .main-page {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  
  header {
    background-color: #333;
    color: #fff;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
  }
  
  nav ul {
    display: flex;
    list-style: none;
    gap: 1rem;
  }
  
  nav a {
    color: #fff;
    text-decoration: none;
  }
  
  .user-info {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  
  main {
    flex-grow: 1;
    padding: 2rem;
  }
  </style>