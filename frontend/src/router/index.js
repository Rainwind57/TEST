import { createRouter, createWebHistory } from 'vue-router'
import QuoteView from '../views/QuoteView.vue'
import FactorView from '../views/FactorView.vue'
import RegressionView from '../views/RegressionView.vue'
import FactorRegressionView from '../views/FactorRegressionView.vue'
import SelectView from '../views/SelectView.vue'
import BacktestView from '../views/BacktestView.vue'
import PortfolioView from '../views/PortfolioView.vue'
import StrategiesView from '../views/StrategiesView.vue'
import MLView from '../views/MLView.vue'
import MonitorView from '../views/MonitorView.vue'
import OptimizeView from '../views/OptimizeView.vue'
import PortfolioOptView from '../views/PortfolioOptView.vue'

const routes = [
  { path: '/', redirect: '/quote' },
  { path: '/quote', name: 'quote', component: QuoteView, meta: { label: '行情' } },
  { path: '/factor', name: 'factor', component: FactorView, meta: { label: '因子' } },
  { path: '/regression', name: 'regression', component: RegressionView, meta: { label: '回归' } },
  { path: '/factor-regression', name: 'factor-regression', component: FactorRegressionView, meta: { label: '多因子回归' } },
  { path: '/select', name: 'select', component: SelectView, meta: { label: '选股' } },
  { path: '/backtest', name: 'backtest', component: BacktestView, meta: { label: '分层回测' } },
  { path: '/portfolio', name: 'portfolio', component: PortfolioView, meta: { label: '模拟盘' } },
  { path: '/strategies', name: 'strategies', component: StrategiesView, meta: { label: '策略中心' } },
  { path: '/ml', name: 'ml', component: MLView, meta: { label: '机器学习' } },
  { path: '/monitor', name: 'monitor', component: MonitorView, meta: { label: '盯盘调度' } },
  { path: '/optimize', name: 'optimize', component: OptimizeView, meta: { label: '参数寻优' } },
  { path: '/portfolio-opt', name: 'portfolio-opt', component: PortfolioOptView, meta: { label: '组合优化' } }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
