'use client';

import { useState } from 'react';
import TaskInput from '@/components/TaskInput';
import TaskList, { Task } from '@/components/TaskList';

export default function Home() {
  const [tasks, setTasks] = useState<Task[]>([]);

  const handleAddTask = (title: string) => {
    const newTask: Task = {
      id: crypto.randomUUID(),
      title,
      completed: false,
    };
    setTasks([...tasks, newTask]);
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 py-12 px-4">
      <main className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-800 dark:text-white mb-8">
          Task Tracker
        </h1>

        <div className="flex flex-col items-center gap-8">
          <TaskInput onAddTask={handleAddTask} />
          <TaskList tasks={tasks} />
        </div>

        <p className="text-center text-sm text-gray-400 mt-12">
          Built to test the Claude Code GitHub Action
        </p>
      </main>
    </div>
  );
}
