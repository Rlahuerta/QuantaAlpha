import React, { useState, useEffect, useCallback } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import {
  TrendingUp,
  Activity,
  DollarSign,
  RefreshCw,
  Wifi,
  WifiOff,
  BarChart3,
  List,
  AlertCircle,
} from 'lucide-react';
import {
  getLiveStatus,
  getLivePositions,
  getLivePnl,
  getLiveSignal,
  type LiveStatus,
  type LivePositions,
  type PnlSnapshot,
  type LiveSignal,
} from '@/services/api';

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function fmtPnl(v: number): string {
  const sign = v >= 0 ? '+' : '';
  return `${sign}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  return iso.length > 10 ? iso.slice(0, 10) : iso;
}

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────

const KpiCard: React.FC<{
  icon: React.ElementType;
  label: string;
  value: string;
  sub?: string;
  positive?: boolean | null;
}> = ({ icon: Icon, label, value, sub, positive }) => (
  <div className="glass rounded-xl p-4 card-hover flex flex-col gap-1">
    <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
      <Icon className="h-4 w-4" />
      {label}
    </div>
    <div
      className={`text-2xl font-bold ${
        positive === true
          ? 'text-green-400'
          : positive === false
          ? 'text-red-400'
          : ''
      }`}
    >
      {value}
    </div>
    {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
  </div>
);

const PnlChart: React.FC<{ history: PnlSnapshot[] }> = ({ history }) => {
  if (history.length === 0)
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground text-sm">
        No P&amp;L history yet
      </div>
    );

  const data = history.map((s) => ({
    date: fmtDate(s.date),
    cumulative: s.cumulative_pnl,
    daily: s.daily_pnl,
  }));

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.cumulative)));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: '#888' }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#888' }}
          tickLine={false}
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          domain={[-maxAbs * 1.1, maxAbs * 1.1]}
        />
        <Tooltip
          contentStyle={{ background: '#1e1e2e', border: '1px solid #333', borderRadius: 8 }}
          formatter={(v: number) => [`$${v.toLocaleString()}`, 'Cumulative P&L']}
          labelStyle={{ color: '#aaa', fontSize: 11 }}
        />
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
        <Area
          type="monotone"
          dataKey="cumulative"
          stroke="#22c55e"
          strokeWidth={2}
          fill="url(#pnlGrad)"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

const PositionsTable: React.FC<{ positions: Record<string, number> }> = ({ positions }) => {
  const entries = Object.entries(positions).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0)
    return (
      <div className="text-muted-foreground text-sm text-center py-6">
        No open positions
      </div>
    );
  return (
    <div className="overflow-auto max-h-56">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border/30 text-muted-foreground text-xs">
            <th className="text-left py-2 px-3">Ticker</th>
            <th className="text-right py-2 px-3">Shares</th>
            <th className="text-right py-2 px-3">Direction</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([ticker, shares]) => (
            <tr key={ticker} className="border-b border-border/10 hover:bg-white/5">
              <td className="py-1.5 px-3 font-mono font-medium">{ticker}</td>
              <td className="py-1.5 px-3 text-right">{shares.toLocaleString()}</td>
              <td className="py-1.5 px-3 text-right">
                <Badge variant={shares > 0 ? 'success' : 'destructive'} className="text-xs">
                  {shares > 0 ? 'LONG' : 'SHORT'}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const OrdersLog: React.FC<{ signal: LiveSignal | null }> = ({ signal }) => {
  if (!signal || signal.orders.length === 0)
    return (
      <div className="text-muted-foreground text-sm text-center py-6">
        No orders in latest signal
      </div>
    );
  return (
    <div className="overflow-auto max-h-56">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border/30 text-muted-foreground text-xs">
            <th className="text-left py-2 px-3">Ticker</th>
            <th className="text-right py-2 px-3">Shares</th>
            <th className="text-right py-2 px-3">Action</th>
            <th className="text-left py-2 px-3 hidden md:table-cell">Reason</th>
          </tr>
        </thead>
        <tbody>
          {signal.orders.map((o, i) => (
            <tr key={i} className="border-b border-border/10 hover:bg-white/5">
              <td className="py-1.5 px-3 font-mono font-medium">{o.ticker}</td>
              <td className="py-1.5 px-3 text-right">{Math.abs(o.shares)}</td>
              <td className="py-1.5 px-3 text-right">
                <Badge
                  variant={o.action === 'BUY' ? 'success' : 'destructive'}
                  className="text-xs"
                >
                  {o.action}
                </Badge>
              </td>
              <td className="py-1.5 px-3 text-muted-foreground text-xs hidden md:table-cell truncate max-w-[160px]">
                {o.reason}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ─────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────

export const LiveTradingPage: React.FC = () => {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [positions, setPositions] = useState<LivePositions | null>(null);
  const [pnlHistory, setPnlHistory] = useState<PnlSnapshot[]>([]);
  const [signal, setSignal] = useState<LiveSignal | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statusRes, posRes, pnlRes, sigRes] = await Promise.all([
        getLiveStatus(),
        getLivePositions(),
        getLivePnl(90),
        getLiveSignal(),
      ]);
      if (statusRes.success) setStatus(statusRes.data ?? null);
      if (posRes.success) setPositions(posRes.data ?? null);
      if (pnlRes.success) setPnlHistory(pnlRes.data?.history ?? []);
      if (sigRes.success) setSignal(sigRes.data ?? null);
      setLastRefresh(new Date());
    } catch (e: any) {
      setError(e.message ?? 'Failed to load live data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30_000); // auto-refresh every 30 s
    return () => clearInterval(interval);
  }, [refresh]);

  // ── KPI values ──
  const totalPnl = pnlHistory.length > 0 ? pnlHistory[pnlHistory.length - 1].cumulative_pnl : 0;
  const lastDailyPnl = pnlHistory.length > 0 ? pnlHistory[pnlHistory.length - 1].daily_pnl : 0;
  const numPositions = positions?.num_positions ?? 0;
  const signalDate = signal?.date ?? status?.last_signal_date ?? null;
  const lastPnlDate = pnlHistory.length > 0 ? pnlHistory[pnlHistory.length - 1].date : null;

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Live Trading</h2>
          <p className="text-muted-foreground text-sm mt-0.5">
            Paper trading · IBKR TWS · S&amp;P 500 universe
          </p>
        </div>
        <div className="flex items-center gap-3">
          {status && (
            <Badge
              variant={status.ibkr_connected ? 'success' : 'default'}
              className="gap-1"
            >
              {status.ibkr_connected ? (
                <Wifi className="h-3 w-3" />
              ) : (
                <WifiOff className="h-3 w-3" />
              )}
              {status.ibkr_connected ? 'IBKR Connected' : status.dry_run ? 'Dry Run' : 'Disconnected'}
            </Badge>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={refresh}
            disabled={loading}
            className="gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 border border-destructive/30 px-4 py-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          icon={DollarSign}
          label="Cumulative P&L"
          value={fmtPnl(totalPnl)}
          sub={`Last day: ${fmtPnl(lastDailyPnl)}`}
          positive={totalPnl > 0 ? true : totalPnl < 0 ? false : null}
        />
        <KpiCard
          icon={Activity}
          label="Open Positions"
          value={String(numPositions)}
          sub={`Capital: ${positions?.capital ? `$${(positions.capital / 1e6).toFixed(1)}M` : '—'}`}
        />
        <KpiCard
          icon={BarChart3}
          label="P&L Days"
          value={String(pnlHistory.length)}
          sub={`Latest: ${fmtDate(lastPnlDate)}`}
        />
        <KpiCard
          icon={List}
          label="Last Signal"
          value={fmtDate(signalDate)}
          sub={signal ? `${signal.orders.length} orders · ${signal.scores_count} scores` : '—'}
        />
      </div>

      {/* P&L chart */}
      <Card className="glass card-hover">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-green-400" />
            Cumulative P&amp;L
          </CardTitle>
        </CardHeader>
        <CardContent>
          <PnlChart history={pnlHistory} />
        </CardContent>
      </Card>

      {/* Positions + Orders */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card className="glass card-hover">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              Current Positions
              {numPositions > 0 && (
                <Badge variant="default" className="ml-auto text-xs">
                  {numPositions}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <PositionsTable positions={positions?.positions ?? {}} />
          </CardContent>
        </Card>

        <Card className="glass card-hover">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <List className="h-4 w-4 text-warning" />
              Latest Orders
              <span className="ml-auto text-xs text-muted-foreground">
                {signalDate ? `Signal date: ${fmtDate(signalDate)}` : 'No signal yet'}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <OrdersLog signal={signal} />
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      {lastRefresh && (
        <p className="text-xs text-muted-foreground text-right">
          Last refreshed: {lastRefresh.toLocaleTimeString()} · auto-refresh every 30 s
        </p>
      )}
    </div>
  );
};
