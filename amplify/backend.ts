import { defineBackend, defineHttpDataSource } from '@aws-amplify/backend';
import { data } from './data/resource';

/**
 * @see https://docs.amplify.aws/react/build-a-backend/ to add storage, functions, and more
 */
export const PythonBackend = defineHttpDataSource({
  name: 'PythonBackend',
  endpoint: 'https://d398t63webqfja.cloudfront.net',
  authorizationConfig: {
    signingRegion: 'us-east-2',
    signingServiceName: 'elasticbeanstalk',
  },
});

defineBackend({
  data,
  PythonBackend,
});
