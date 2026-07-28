import { createRouter, createWebHistory } from 'vue-router'
import QuoteView from '../views/QuoteView.vue'
import FactorView from '../views/FactorView.vue'
import RegressionView from '../views/RegressionView.vue'
import FactorRegressionView from '../views/FactorRegressionView.vue'
import SelectView from '../views/SelectView.vue'
import BacktestView from '../views/BacktestView.vue'
import PortfolioView from '../views/PortfolioView.vue'

const routes = [
  { path: '/', redirect: '/quote' },
  { path: '/quote', name: 'quote', component: QuoteView, meta: { label: '行情' } },
  { path: '/factor', name: 'factor', component: FactorView, meta: { label: '因子' } },
  { path: '/regression', name: 'regression', component: RegressionView, meta: { label: '回归' } },
  { path: '/factor-regression', name: 'factor-regression', component: FactorRegressionView, meta: { label: '多因子回归' } },
  { path: '/select', name: 'select', component: SelectView, meta: { label: '选股' } },
  { path: '/backtest', name: 'backtest', component: BacktestView, meta: { label: '分层回测' } },
  { path: '/portfolio', name: 'portfolio', component: PortfolioView, meta: { label: '模拟盘' } }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
