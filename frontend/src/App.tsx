import './App.css';

import { Button, Divider, FileInput, Group, NativeSelect, Progress } from '@mantine/core';
import React, { useState } from 'react';

import { notifications } from '@mantine/notifications';

function App() {
  const [value, setValue] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);



  return (
    <>
      <Group grow style={{ marginLeft: 10, marginRight: 10, marginTop: 10 }}>
        <NativeSelect label="Data metrics" description="Choose which metrics to output" data={['React', 'Angular', 'Vue']} />
        <NativeSelect label="Model" description="Choose which model to use  " data={['React', 'Angular', 'Vue']} />
      </Group>
      <Divider my="sm" />
      <FileInput
        label="Upload Data"
        description="Select data files to upload"
        placeholder="Select file.."
        style={{ marginLeft: 10, marginRight: 10, marginTop: 10 }}
        value={value}
        onChange={setValue}

      />
      <Divider my="sm" />
      <Group>
        <Button style={{ marginLeft: 10, marginRight: 10, marginTop: 10 }} onClick={() => {
          if (value !== null) {
            setProgress(0);
            const reader = new FileReader();
            reader.onprogress = (evt) => onProgressCallback(evt, setProgress);
            reader.onload = (evt) => onLoadCallback(evt);
            reader.readAsArrayBuffer(value);
          }
        }}>Start</Button >
      </Group>
      <Progress size="sm" value={progress} style={{ marginLeft: 10, marginRight: 10, marginTop: 10 }} />
    </>
  );
}

function onLoadCallback(event: any) {
  if (event.target) {
    const buffer = event.target.result;
    if (buffer instanceof ArrayBuffer) {
      const fd = new FormData();
      fd.append('binary_data', new Blob([buffer]));
      try {
        fetch('http://127.0.0.1:5000/file-input', {
          method: 'POST',
          body: fd
        })
      }
      catch (e) {
        console.log(e);
      }
    }
  }
}

function onProgressCallback(event: any, setProgress: any) {
  const p = (event.loaded / event.total) * 100;
  setProgress(p);
  if (p === 100) {
    notifications.show({
      title: `Success`,
      message: `Successfully uploaded data.`,
      color: 'green',
      autoClose: 5000
    })
  }
}

export default App;
