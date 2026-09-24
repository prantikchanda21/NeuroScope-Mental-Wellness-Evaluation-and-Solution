// Netlify Functions run as short-lived serverless invocations, not a persistent
// `node server.ts` process — so the Express app has to be adapted to that model
// instead of just being started with app.listen(). serverless-http does that
// adaptation: it wraps the same Express app (shared from server-app.ts) so each
// Netlify Function invocation is handled like one incoming Express request.
//
// netlify.toml redirects /api/* to /.netlify/functions/api/:splat, which means
// Netlify hands this function a path like /.netlify/functions/api/assess. Our
// Express routes are defined as /api/assess etc., so that prefix is rewritten
// back to /api/... below before handing the request to Express.
import type { Handler, HandlerEvent, HandlerContext } from '@netlify/functions';
import serverless from 'serverless-http';
import app from '../../server-app';

const expressHandler = serverless(app);

export const handler: Handler = async (event: HandlerEvent, context: HandlerContext) => {
  const rewrittenPath = event.path.replace(/^\/\.netlify\/functions\/api/, '/api') || '/api';

  const rewrittenEvent = {
    ...event,
    path: rewrittenPath,
  };

  return expressHandler(rewrittenEvent as any, context as any) as any;
};
