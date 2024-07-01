import { createRouter, createWebHistory } from 'vue-router';
import LoginPage from '../views/login.vue';
import MainPage from '../views/main.vue';

const routes = [
  {
    path: '/',
    name: 'login',
    component: LoginPage
  },
  {
    path: '/main',
    name: 'main',
    component: MainPage,
    meta: {
      requiresAuth: true
    }
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach((to, from, next) => {
  // 在这里添加登录验证逻辑
  if (to.meta.requiresAuth && !isLoggedIn()) {
    next('/');
  } else {
    next();
  }
});

export default router;