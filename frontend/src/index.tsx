import './index.css';

import * as ReactDOM from 'react-dom';

import App from './App';
import { MantineProvider } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { Provider } from 'react-redux';
import React from 'react';
import { store } from './redux/store';

ReactDOM.render(
  <MantineProvider withNormalizeCSS withGlobalStyles>
    <Provider store={store}>
      <Notifications limit={3} />
      <App />
    </Provider>
  </MantineProvider>,
  document.getElementById('root')
);  